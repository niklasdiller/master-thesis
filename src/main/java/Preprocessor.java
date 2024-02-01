package main.java;

import org.json.simple.parser.ParseException;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.sql.*;
import java.util.Properties;

public class Preprocessor {

    private Settings settings;

    private ModelTrainer trainer;

    private int periodMinutes;

    private int parkingID;

    public Connection conn;

    /**
     * Create a model trainer
     *
     * @param settings Contains all settings to run training pipeline
     */
    public Preprocessor(Settings settings, ModelTrainer trainer) throws IOException, ParseException {
        this.settings = settings;
        this.periodMinutes = settings.periodMinutes;
        this.parkingID = settings.parkingId;
        this.trainer = trainer;
    }

    private void saveTable(Table table, int pID, int perMin, boolean shift24h) throws SQLException {
        //add pID+perMin+24hShift identifier to table
        String context = "pID" + pID + "_perMin" + perMin;
        if (shift24h) context += "_24h";

        PreparedStatement ps = trainer.conn.prepareStatement("" +
                "INSERT INTO " + settings.tableName + " (" +
                "temp, humidity, weekday, month, year, time_Slot, previous_Occupancy, occupancy, context) " +
                "VALUES (?,?,?,?,?,?,?,?,?);");

        System.out.println(table.rowCount() + " Rows in Table");

        for (int i = 0; i < table.rowCount(); i++) {
            ps.setDouble(1, table.row(i).getDouble("Temp"));
            ps.setDouble(2, table.row(i).getDouble("Humidity"));
            ps.setInt(3, table.row(i).getInt("weekDay"));
            ps.setInt(4, table.row(i).getInt("month"));
            ps.setInt(5, table.row(i).getInt("year"));
            ps.setInt(6, table.row(i).getInt("timeSlot"));
            ps.setDouble(7, table.row(i).getDouble("previousOccupancy"));
            ps.setDouble(8, table.row(i).getDouble("occupancy"));
            ps.setString(9, context);

            ps.addBatch();

            if (i % 1000 == 0 || i == table.rowCount() - 1) { //execute every 1000 rows
                ps.executeBatch();
            }
        }
        ps.close();
        System.out.println("Data saved to DB as " + context);
    }

    private Properties changeValues(String settingsPath, int pID, int perMin) {
        try {
            FileInputStream in = new FileInputStream("src/" + settingsPath);
            Properties props = new Properties();
            props.load(in);
            in.close();

            FileOutputStream out = new FileOutputStream("src/" + settingsPath);
            props.setProperty("parkingId", String.valueOf(pID));
            props.setProperty("periodMinutes", String.valueOf(perMin));

            props.store(out, null);
            out.close();
            return props;

        } catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    public static void main(String[] args) {
        try {
            String settingsPath = "main/java/preprocessedDB.properties";
            InputStream input = ModelTrainer.class.getClassLoader().getResourceAsStream(settingsPath);
            Properties props = new Properties();
            props.load(input);
            Settings settings = new Settings(settingsPath, props);
            ModelTrainer trainer = new ModelTrainer(settings);
            Preprocessor prep = new Preprocessor(settings, trainer);
            trainer.createDBConnection();

            int pID_val;
            int perMin_val;
            boolean shift24h = false;

            //Parking Lot
            for (int pID = 0; pID <= trainer.parkingLotMap.size() - 1; pID++) {
                pID_val = trainer.parkingLotMap.get(pID);

                //Period Minutes
                for (int perMin = 0; perMin <= trainer.periodMinuteMap.size() - 1; perMin++) {
                    perMin_val = trainer.periodMinuteMap.get(perMin).get(0);

                    //set flag for 24h occupancy prediction used in preprocessing
                    if (perMin == 3) shift24h = true;

                    props = prep.changeValues(settingsPath, pID_val, perMin_val);
                    settings = new Settings(settingsPath, props);
                    trainer = new ModelTrainer(settings);
                    prep = new Preprocessor(settings, trainer);
                    ResultSet rs = trainer.queryDB();
                    Table tableData = trainer.preprocessing(rs, shift24h);
                    prep.saveTable(tableData, pID_val, perMin_val, shift24h);
                    shift24h = false; //reset flag
                }
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
