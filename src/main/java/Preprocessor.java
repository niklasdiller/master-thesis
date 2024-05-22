package main.java;

import org.json.simple.parser.ParseException;
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

    /**
     * Create a model trainer
     *
     * @param settings Contains all settings to run training pipeline
     */
    public Preprocessor(Settings settings, ModelTrainer trainer) throws IOException, ParseException {
        this.settings = settings;
        this.trainer = trainer;
    }

    private void saveTable(Table table, int pID, int winSize, boolean shift24h) throws SQLException {

        System.out.println("Saving table with pID=" + pID + " and winSize=" + winSize + " to DB.");

        PreparedStatement ps = trainer.conn.prepareStatement("" +
                "INSERT INTO " + settings.tableName + " (" +
                "temp, humidity, weekday, month, year, time_Slot, previous_Occupancy, occupancy, pID, window_size," +
                " shift24h, period_start_time) " + "VALUES (?,?,?,?,?,?,?,?,?,?,?,?);");

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
            ps.setInt(9, pID);
            ps.setInt(10, winSize);
            ps.setBoolean(11, shift24h); // Indicates if a 24h Shift was used for this data entry
            ps.setString(12, table.row(i).getDateTime("periodStartTime").toString()); // Time of start of this period

            ps.addBatch();

            if (i % 1000 == 0 || i == table.rowCount() - 1) { //execute every 1000 rows
                ps.executeBatch();
            }
        }
        ps.close();
        System.out.println("Data with pID=" + pID + " and winSize=" + winSize +" saved to DB.");
    }

    private Properties changeValues(String settingsPath, int pID, int winSize) {
        try {
            FileInputStream in = new FileInputStream("src/" + settingsPath);
            Properties props = new Properties();
            props.load(in);
            in.close();

            FileOutputStream out = new FileOutputStream("src/" + settingsPath);
            props.setProperty("parkingId", String.valueOf(pID));
            props.setProperty("windowSize", String.valueOf(winSize));

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
            String settingsPath = "main/java/preprocess.properties";
            InputStream input = ModelTrainer.class.getClassLoader().getResourceAsStream(settingsPath);
            Properties props = new Properties();
            props.load(input);
            Settings settings = new Settings(settingsPath, props);
            ModelTrainer trainer = new ModelTrainer(settings);
            Preprocessor prep = new Preprocessor(settings, trainer);
            trainer.createDBConnection();

            int pID_val;
            int winSize_val;
            boolean shift24h = false;

            //Parking Lot
            for (int pID = 0; pID <= trainer.parkingLotMap.size() - 1; pID++) {
                pID_val = trainer.parkingLotMap.get(pID);

                //Window Size
//                for (int winSize = 0; winSize <= trainer.windowSizeMap.size() - 1; winSize++) {
//                    winSize_val = trainer.windowSizeMap.get(winSize).get(0); //TODO: Uncomment lines
                winSize_val = 5;

                //set flag for 24h occupancy prediction used in preprocessing
//                    if (winSize == 3) shift24h = true; //TODO Uncomment

                props = prep.changeValues(settingsPath, pID_val, winSize_val);
                settings = new Settings(settingsPath, props);
                trainer = new ModelTrainer(settings);
                prep = new Preprocessor(settings, trainer);

                ResultSet rs = trainer.queryDB();
                Table tableData = trainer.preprocessing(rs, shift24h);
                prep.saveTable(tableData, pID_val, winSize_val, shift24h);
                rs.getStatement().close(); // closes the resource
                shift24h = false; //resets flag
//                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
