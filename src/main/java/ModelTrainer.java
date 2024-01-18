package main.java;

import org.json.simple.parser.ParseException;
import tech.tablesaw.io.DataFrameReader;
import tech.tablesaw.io.ReaderRegistry;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;
import weka.core.*;

import java.io.*;
import java.sql.*;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.*;

import tech.tablesaw.api.*;
import tech.tablesaw.selection.Selection;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class ModelTrainer implements Serializable {

    /**
     * list of all possible attributes
     */
    private ArrayList<String> occupancyPredictAttributes = new ArrayList<String>() {{
        add("Temp");
        add("Humidity");
        add("weekDay");
        add("month");
        add("year");
        add("timeSlot");
        add("previousOccupancy");
        add("occupancy");
    }};

    /**
     * attribute indexes for actual model
     */
    private ArrayList<Integer> attributeIndexes = new ArrayList<>();

    /**
     * Map for attribute names
     **/
    private Map<Integer, String> attributesNamesMap = new HashMap<Integer, String>();

    /**
     * All settings specified in properties file
     */
    private Settings settings;

    /**
     * Path to properties file
     */
    private String settingsPath;

    /**
     * Database connection
     */
    private static Connection conn;

    /**
     * Period in minutes
     */
    private int periodMinutes;

    /**
     * The training data gathered so far.
     */
    private Instances m_Train_Data;

    /**
     * The testing data gathered so far.
     */
    private Instances m_Test_Data;

    /**
     * The Decision Tree classifier.
     */
    private M5P m_DecisionTreeClassifier = new M5P();
    /**
     * Decision Tree accuracy variables
     */
    private int correctPredictedDT;
    private double MAE_DT;
    private double MSE_DT;

    /**
     * The Random Forest classifier.
     */
    private RandomForest m_RandomForestClassifier = new RandomForest();

    /**
     * Random Forest accuracy variables
     */
    private int correctPredictedRF;
    private double MAE_RF;
    private double MSE_RF;

    /**
     * The Linear Regression classifiert
     **/
    private LinearRegression m_LinearRegressionClassifier = new LinearRegression();

    /**
     * Linear Regression accuracy variables
     */
    private int correctPredictedLR;
    private double MAE_LR;
    private double MSE_LR;

    /**
     * The k-Nearest Neighbors classifier.
     */
    private IBk m_KNNClassifier = new IBk();

    /**
     * k-Nearest Neighbors accuracy variables
     */
    private int correctPredictedKNN;
    private double MAE_KNN;
    private double MSE_KNN;

    /**
     * Map for classifier choice
     **/
    private Map<Integer, Classifier> classifierMap = new HashMap<Integer, Classifier>();

    /**
     * Map for classifier names
     **/
    private Map<Integer, String> classifierNamesMap = new HashMap<Integer, String>();

    /**
     * Map for periodMinute values
     **/
    private Map<Integer, List<Integer>> periodMinuteMap = new HashMap<>();

    /**
     * Map for parkingLot values
     **/
    private Map<Integer, Integer> parkingLotMap = new HashMap<>();

    /**
     * Create a model trainer
     *
     * @param settings Contains all settings to run training pipeline
     */
    public ModelTrainer(Settings settings) throws IOException, ParseException {
        this.settings = settings;
        this.periodMinutes = settings.periodMinutes;

        String nameOfDataset = "ParkingOccupancyProblem";

        ArrayList<Attribute> attributes = new ArrayList<>();
        if (settings.attributesData.isEmpty()) {
            for (int i = 0; i < occupancyPredictAttributes.size() - 1; i++) {
                attributes.add(new Attribute(occupancyPredictAttributes.get(i)));
                this.attributeIndexes.add(i);
            }
        } else {
            for (int i : settings.attributesData) {
                attributes.add(new Attribute(occupancyPredictAttributes.get(i)));
                this.attributeIndexes.add(i);
            }
        }
        Attribute occupancyAttribute = new Attribute(occupancyPredictAttributes.get(occupancyPredictAttributes.size() - 1));
        attributes.add(occupancyAttribute);

        int targetAttributIndex = attributes.indexOf(occupancyAttribute);

        // create dataset with initial capacity of 10000
        m_Train_Data = new Instances(nameOfDataset, attributes, 10000);
        // add label at index targetAttributIndex of output attributes
        m_Train_Data.setClassIndex(targetAttributIndex);

        m_Test_Data = new Instances(nameOfDataset, attributes, 10000);
        m_Test_Data.setClassIndex(targetAttributIndex);

        // fill a map with classifiers
        this.classifierMap.put(0, m_DecisionTreeClassifier);
        this.classifierMap.put(1, m_RandomForestClassifier);
        this.classifierMap.put(2, m_LinearRegressionClassifier);
        this.classifierMap.put(3, m_KNNClassifier);

        // fill a name map with classifier names
        this.classifierNamesMap.put(0, "Decision Tree");
        this.classifierNamesMap.put(1, "Random Forest");
        this.classifierNamesMap.put(2, "Linear Regression");
        this.classifierNamesMap.put(3, "K-Nearest Neighbours");

        // fill an attributes map with attribute names
        this.attributesNamesMap.put(0, "temperature");
        this.attributesNamesMap.put(1, "humidity");
        this.attributesNamesMap.put(2, "day of the week");
        this.attributesNamesMap.put(3, "month");
        this.attributesNamesMap.put(4, "year");
        this.attributesNamesMap.put(5, "previous occupancy");
        this.attributesNamesMap.put(6, "time slot");

        // fill a periodMinuteMap with values for the training pipeline and their corresponding trainingDataSize
        // first value: periodMinutes
        // second value: trainingDataSize to get 1 week of training data total
        // third value: trainingDataSize to get 4 weeks of training data total
        List<Integer> values0 = new ArrayList<>();
        values0.add(10); //window size 10 minutes
        values0.add(1008); // 10080 minutes in a week -> 1008 10min slots in a week
        values0.add(4032); // 40320 minutes in 4 weeks -> 4032 10min slots in 4 weeks
        this.periodMinuteMap.put(0, values0);
        List<Integer> values1 = new ArrayList<>();
        values1.add(30);
        values1.add(336);
        values1.add(1344);
        this.periodMinuteMap.put(1, values1);
        List<Integer> values2 = new ArrayList<>();
        values2.add(60);
        values2.add(163);
        values2.add(672);
        this.periodMinuteMap.put(2, values2);
        List<Integer> values3 = new ArrayList<>();
        values3.add(60);
        values3.add(192); //to get values for 24h after the last instance
        values3.add(696);
        this.periodMinuteMap.put(3, values3);
//        values.clear();

        // fill a parkingLot Map with the corresponding Parking Lot IDs
        this.parkingLotMap.put(0, 38);
        this.parkingLotMap.put(1, 634);

        // hyperparameter, for RF and KNN
        this.m_RandomForestClassifier.setMaxDepth(settings.randomForestMaxDepth);
        this.m_KNNClassifier.setKNN(settings.kNeighbours);
    }

    /**
     * Create a connection to the database
     *
     * @throws SQLException
     */
    private void createDBConnection() throws SQLException {
        Properties props = new Properties();
        props.setProperty("user", settings.dbUsername);
        props.setProperty("password", settings.dbPassword);
        conn = DriverManager.getConnection(settings.dbUri, props);

        System.out.println("Database connection established.");
    }

    /**
     * Get preprocessed data from previously existing table in db
     *
     * @return A ResultSet containing the matching rows
     */
    private ResultSet queryDB() throws SQLException {
        System.out.println("Querying preprocessed data...");
        System.out.println("Data from table " + settings.preprocessTable + settings.parkingId);
        String query = "SELECT * FROM public.\"" + settings.preprocessTable + settings.parkingId +
                "\" ORDER BY arrival_unix_seconds ASC LIMIT " + settings.trainingDataSize + ";";
        Statement st = conn.createStatement();
        return st.executeQuery(query);
    }

    /**
     * Create a new instance based on features calculated from windowData
     *
     * @param rowData Row input data
     * @return Instance with calculated features and label
     * @throws Exception
     */
    private Instance createInstance(Row rowData) throws Exception {
        Instance instance = new DenseInstance(this.m_Train_Data.numAttributes());
        instance.setDataset(this.m_Train_Data);

        for (int attribute : this.attributeIndexes) {
            if (attribute == 0 || attribute == 1 || attribute == 6) {
                instance.setValue(this.m_Train_Data.attribute(occupancyPredictAttributes.get(attribute)),
                        rowData.getDouble(occupancyPredictAttributes.get(attribute)));
            } else if (attribute == 2 || attribute == 3 || attribute == 4 || attribute == 5) {
                instance.setValue(this.m_Train_Data.attribute(occupancyPredictAttributes.get(attribute)),
                        rowData.getInt(occupancyPredictAttributes.get(attribute)));
            }
        }
        // target is proceed separately, because attributesData contains only attributes
        instance.setValue(this.m_Train_Data.attribute(occupancyPredictAttributes.get(occupancyPredictAttributes.size() - 1)),
                rowData.getDouble(occupancyPredictAttributes.get(occupancyPredictAttributes.size() - 1)));

        return instance;
    }

    /**
     * Convert the DB result to instances
     *
     * @param table from preprocessing
     */
    private void saveQueryAsInstances(Table table) throws Exception {
        Instance instance;

        int i = 0;
        int validSize = (int) (settings.trainProp * 10);

        if (table.isEmpty())
            throw new NullPointerException("The table is empty");
        int currentRowIndex = 0;
        Row currentRow = table.row(currentRowIndex);

        if (settings.trainTestStrategy == 0) {
            // to mix train and test data
            while (currentRowIndex < table.rowCount()) {
                instance = createInstance(currentRow);

                if (i < validSize) {
                    m_Train_Data.add(instance);
                    i++;
                } else if (i < 10) {
                    m_Test_Data.add(instance);
                    i++;
                    if (i == 10) {
                        i = 0;
                    }
                }
                currentRowIndex++;
                currentRow = table.row(currentRowIndex);
            }
        } else {
            // to extract test data after train data, not to mix
            double validSizeTest = this.settings.trainProp * table.rowCount();
            while (currentRowIndex < table.rowCount()) {
                instance = createInstance(currentRow);

                if (currentRowIndex < validSizeTest) {
                    m_Train_Data.add(instance);
                } else {
                    m_Test_Data.add(instance);
                }

                currentRowIndex++;
                currentRow = table.row(currentRowIndex);
            }
        }

        System.out.println("Converted data to instances.");
    }

    /**
     * Save the model and all its parameters to the DB
     *
     * @throws IOException
     * @throws SQLException
     */
    //@SuppressWarnings("SqlResolve")
    private void saveModelToDB() throws IOException, SQLException {

        //Created Time
        Instant instant = Instant.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd.MM.yyy, HH:mm:ss ")
                .withZone(ZoneId.systemDefault());
        String formattedInstant = formatter.format(instant);

        // preparation for saving
        String slotIDsString = "", classifierNamesString = "", attributesString = "";

        if (settings.slotsIDData.isEmpty()) {
            slotIDsString += "all IDs";
        } else {
            for (int id : settings.slotsIDData) {
                slotIDsString += (id + " ");
            }
        }

        //Classifier
        if (settings.classifiersData.isEmpty()) { //multiple Classifiers in same model not possible
            System.err.println("Classifier cannot be empty!");
            ;
        } else {
            for (int classifierNumber : settings.classifiersData) {
                classifierNamesString = (classifierNamesMap.get(classifierNumber));
            }
        }

        //Modelsize
        int modelSize = 0;

        if (settings.attributesData.isEmpty()) {
            System.err.println("Attributes cannot be empty!"); //empty case handled through all selected
        } else {
            for (int attributNumber : settings.attributesData) {
                attributesString += (attributesNamesMap.get(attributNumber) + ", ");
            }
            attributesString = attributesString.substring(0, attributesString.length() - 2); //remove last ", "
        }

        String accuracyInfoDTString = "no classifier", accuracyInfoRFString = "no classifier",
                accuracyInfoLRString = "no classifier", accuracyInfoKNNString = "no classifier";
        List<Integer> listForClassifierIndexes = new ArrayList<Integer>();
        if (settings.classifiersData.isEmpty()) {
            listForClassifierIndexes.add(0);
            listForClassifierIndexes.add(1);
            listForClassifierIndexes.add(2);
            listForClassifierIndexes.add(3);
        } else {
            for (int i = 0; i < settings.classifiersData.size(); i++) {
                listForClassifierIndexes.add(settings.classifiersData.get(i));
            }
        }
        for (int i = 0; i < listForClassifierIndexes.size(); i++) {
            if (listForClassifierIndexes.get(i) == 0) {
                accuracyInfoDTString = "Correctly predicted: "
                        + (double) Math.round((correctPredictedDT / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%"
                        + " MAE: " + (double) Math.round(MAE_DT / (double) m_Test_Data.size() * 100) / 100
                        + " MSE: " + (double) Math.round(MSE_DT / (double) m_Test_Data.size() * 100) / 100;
            } else if (listForClassifierIndexes.get(i) == 1) {
                accuracyInfoRFString = "Correctly predicted: "
                        + (double) Math.round((correctPredictedRF / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%"
                        + " MAE: " + (double) Math.round(MAE_RF / (double) m_Test_Data.size() * 100) / 100
                        + " MSE: " + (double) Math.round(MSE_RF / (double) m_Test_Data.size() * 100) / 100;
            } else if (listForClassifierIndexes.get(i) == 2) {
                accuracyInfoLRString = "Correctly predicted: "
                        + (double) Math.round((correctPredictedLR / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%"
                        + " MAE: " + (double) Math.round(MAE_LR / (double) m_Test_Data.size() * 100) / 100
                        + " MSE: " + (double) Math.round(MSE_LR / (double) m_Test_Data.size() * 100) / 100;
            } else if (listForClassifierIndexes.get(i) == 3) {
                accuracyInfoKNNString = "Correctly predicted: "
                        + (double) Math.round((correctPredictedKNN / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%"
                        + " MAE: " + (double) Math.round(MAE_KNN / (double) m_Test_Data.size() * 100) / 100
                        + " MSE: " + (double) Math.round(MSE_KNN / (double) m_Test_Data.size() * 100) / 100;
            }
        }

        // saving in database
        System.out.println("Saving model to database...");
        // columns from table to save
        PreparedStatement ps = conn.prepareStatement("" +
                "INSERT INTO " + settings.tableName + " (" +
                "model_name, developer, train_test_strategy, created_time, model_size_in_bytes," +
                "parking_id, training_data_size, period_minutes, slotsIDs," +
                "classifiers, attributes, trainingDataProportion," +
                "accuracyPercent, randomForestMaxDepth, kNeighbours, " +
                "accuracyDT, accuracyRF, accuracyLR, accuracyKNN, decision_tree," +
                "random_forest, linear_regression, k_nearest_neighbors, window_stride) " +
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"); // number of ? has to be the same as the number of columns

        ps.setString(1, settings.modelName);
        ps.setString(2, settings.developer);
        ps.setString(3, settings.trainTestText);
        ps.setString(4, formattedInstant);
        // ModelSize: parameterIndex 5. See down below
        ps.setInt(6, settings.parkingId);
        ps.setInt(7, settings.trainingDataSize);
        ps.setInt(8, settings.periodMinutes);     // periods duration in minutes
        ps.setString(9, slotIDsString);     // ids of parking slots to parse
        ps.setString(10, classifierNamesString);     // classifier names
        ps.setString(11, attributesString); // attributes
        ps.setDouble(12, settings.trainProp); // train part
        ps.setInt(13, settings.accuracyPercent);     // deviation percentage for accuracy calculation
        ps.setInt(14, settings.randomForestMaxDepth);     // max depth for Random Forest Classifier
        ps.setInt(15, settings.kNeighbours);     // number of neighbours for k-Nearest Neighbours Classifier

        // accuracy results
        ps.setString(16, accuracyInfoDTString);
        ps.setString(17, accuracyInfoRFString);
        ps.setString(18, accuracyInfoLRString);
        ps.setString(19, accuracyInfoKNNString);

        // writing trained classifiers (if exist) binary data
        int columnIndexToWrite = 20;
        for (int i = 0; i < 4; i++) {
            if (listForClassifierIndexes.isEmpty() || listForClassifierIndexes.contains(i)) {

                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                ObjectOutputStream out = new ObjectOutputStream(bos);
                if (i == 0) {
                    out.writeObject(this.m_DecisionTreeClassifier);
                } else if (i == 1) {
                    out.writeObject(this.m_RandomForestClassifier);
                } else if (i == 2) {
                    out.writeObject(this.m_LinearRegressionClassifier);
                } else if (i == 3) {
                    out.writeObject(this.m_KNNClassifier);
                }
                out.flush();
                byte[] serializedClassifier = bos.toByteArray();
                modelSize += serializedClassifier.length;
                bos.close();
                ByteArrayInputStream bis = new ByteArrayInputStream(serializedClassifier);
                ps.setBinaryStream(columnIndexToWrite + i, bis, serializedClassifier.length);
                bis.close();
            } else {
                ps.setString(columnIndexToWrite + i, "no classifier");
            }
        }
        ps.setInt(5, modelSize);
        ps.setInt(24, settings.periodMinutes); //Window Stride always same as Window Size (period minutes)

        ps.executeUpdate(); // execution
        ps.close();

        System.out.println("Model saved in database.");
    }

    /**
     * additional function which convert the chosen classifier object to base64 encoded string
     *
     * @param classifierIndex index of classifier to convert
     * @return Classifier encoded in base64 string
     * @throws IOException
     */
    private String classifierToString(int classifierIndex) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        if (classifierIndex == 0) {
            oos.writeObject(this.m_DecisionTreeClassifier);
        } else if (classifierIndex == 1) {
            oos.writeObject(this.m_RandomForestClassifier);
        } else if (classifierIndex == 2) {
            oos.writeObject(this.m_LinearRegressionClassifier);
        } else if (classifierIndex == 3) {
            oos.writeObject(this.m_KNNClassifier);
        }
        oos.close();
        return Base64.getEncoder().encodeToString(baos.toByteArray());
    }

    /**
     * Save encoded base64 string in file
     *
     * @param base64   The content to be saved
     * @param fileName Path where to save the string
     * @throws IOException
     */
    private void saveStringAsFile(String base64, String fileName) throws IOException {
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName));
        oos.writeChars(base64);
        oos.flush();
        oos.close();
        System.out.println("Saved base64 string at location: " + fileName);
    }

    private void testClassifier() throws Exception {
        System.out.println("Testing model...");
        int correctPredicted = 0;
        double meanAbsErr = 0, meanSqErr = 0;
        int percentForAccuracy = settings.accuracyPercent;

        List<Integer> classifiersIndexes = new ArrayList<>();

        if (settings.classifiersData.isEmpty()) {
            for (int i = 0; i < this.classifierMap.size(); i++) {
                classifiersIndexes.add(i);
            }
        } else {
            for (int i : settings.classifiersData) {
                classifiersIndexes.add(i);
            }
        }

        for (int index : classifiersIndexes) {
            correctPredicted = 0;
            meanAbsErr = 0;
            meanSqErr = 0;
            for (Instance instance : m_Test_Data) {
                double value = instance.classValue();
                double prediction = this.classifierMap.get(index).classifyInstance(instance);
                double difference = prediction - value;
                meanAbsErr += Math.abs(difference);
                meanSqErr += difference * difference;
                if (prediction >= value - percentForAccuracy && prediction <= value + percentForAccuracy) {
                    correctPredicted++;
                }
            }
            if (index == 0) {
                correctPredictedDT = correctPredicted;
                MAE_DT = meanAbsErr;
                MSE_DT = meanSqErr;
            } else if (index == 1) {
                correctPredictedRF = correctPredicted;
                MAE_RF = meanAbsErr;
                MSE_RF = meanSqErr;
            } else if (index == 2) {
                correctPredictedLR = correctPredicted;
                MAE_LR = meanAbsErr;
                MSE_LR = meanSqErr;
            } else if (index == 3) {
                correctPredictedKNN = correctPredicted;
                MAE_KNN = meanAbsErr;
                MSE_KNN = meanSqErr;
            }
            System.out.println("\nCorrectly predicted " + classifierNamesMap.get(index) + " "
                    + (double) Math.round((correctPredicted / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%");
            System.out.println(classifierNamesMap.get(index) + " Mean Absolute Error: "
                    + (double) Math.round(meanAbsErr / (double) m_Test_Data.size() * 100) / 100);
            System.out.println(classifierNamesMap.get(index) + " Mean Squared Error: " +
                    (double) Math.round(meanSqErr / (double) m_Test_Data.size() * 100) / 100);
        }
    }

    /**
     * Return processed table of occupancy data
     *
     * @param rs Set of data to process
     * @return Table object
     * @throws Exception
     */
    private Table preprocessing(ResultSet rs, boolean shift24h) throws Exception {
        Table dataAllID = new DataFrameReader(new ReaderRegistry()).db(rs);
        Table data = dataAllID.emptyCopy();

        // if there are special slots to process and predict (e.g. for disabled people)
        if (!this.settings.slotsIDData.isEmpty()) {
            System.out.println("ID of parking slots to parse: ");
            for (int id : settings.slotsIDData) {
                System.out.print(id + " ");
                Table tmpTable = dataAllID.where(dataAllID.longColumn("parking_space_id").isEqualTo(id));
                data.append(tmpTable);
            }
        } else
            data = dataAllID.copy();

        data.setName("Parking data");
        System.out.println("Parking data is imported from DB, data shape is " + data.shape());
        // get number of sensors (unique values)
        int sensorCount = data.column("parking_space_id").countUnique();

        // delete unnecessary columns and sort asc
        data.removeColumns("parking_lot_id", "xml_id", "parking_space_id");
        data.sortAscendingOn("arrival_unix_seconds");

        // seconds to data converting
        String pattern = "dd.MM.yyyy HH:mm:ss"; // pattern to format the date from string
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(pattern); // instance to format the date with pattern

        // getting the first arrival time in "arrival_local_time"
        String startDateString = data.getString(0, "arrival_local_time");
        LocalDateTime START_DATE = LocalDateTime.parse(startDateString, formatter);
        START_DATE = START_DATE.truncatedTo(ChronoUnit.HOURS);     // START_DATE rounded to hours to get the first hour
        if (periodMinutes < 60) {
            int remainder = START_DATE.getMinute() % periodMinutes;
            if (remainder != 0) {
                START_DATE = START_DATE.plusMinutes(periodMinutes - remainder); // Round up to the next N-minute interval
            }
        }
        // getting the last arrival time in "arrival_local_time" and process as START_DATE
        String endDateString = data.getString(data.rowCount() - 1, "arrival_local_time");
        LocalDateTime END_DATE = LocalDateTime.parse(endDateString, formatter);
        END_DATE = END_DATE.truncatedTo(ChronoUnit.HOURS);
        if (periodMinutes < 60) {
            int remainder = END_DATE.getMinute() % periodMinutes;
            if (remainder != 0) {
                END_DATE = END_DATE.plusMinutes(periodMinutes - remainder);
            }
        }

        Table dataWithOccupancy = data.emptyCopy();     // copy of data to filter
        // adding column to concatenate with new rows
        dataWithOccupancy.addColumns(StringColumn.create("periodStart", dataWithOccupancy.rowCount()),
                StringColumn.create("periodEnd", dataWithOccupancy.rowCount()),
                LongColumn.create("occupancySeconds", dataWithOccupancy.rowCount()),
                LongColumn.create("periodStartSeconds"));


        while (START_DATE.isBefore(END_DATE)) {
            dataWithOccupancy.append(filteredByExactPeriod(START_DATE, data, periodMinutes));
            START_DATE = START_DATE.plusMinutes(periodMinutes);
        }

        // removing unnecessary information
        dataWithOccupancy.removeColumns("arrival_unix_seconds", "departure_unix_seconds",
                "arrival_local_time", "departure_local_time");  // removing unnecessary columns

        // processing occupancy sum
        dataWithOccupancy.addColumns(IntColumn.create("occupancySum", dataWithOccupancy.rowCount()));
        Map<String, Integer> occupancySumDataMap = new HashMap<String, Integer>();

        // for every periodStart save the sum or add to sum of occupancy in HashMap
        for (int i = 0; i < dataWithOccupancy.rowCount(); i++) {
            String getKey = dataWithOccupancy.getString(i, "periodStart"); // the time and date
            int getValue = Integer.valueOf(dataWithOccupancy.getString(i, "occupancySeconds")); //occupancy
            if (occupancySumDataMap.containsKey(getKey))
                occupancySumDataMap.put(getKey, getValue + occupancySumDataMap.get(getKey));
            else
                occupancySumDataMap.put(getKey, getValue);
        }

        // and put the occupancy value in "occupancySum" column
        for (int i = 0; i < dataWithOccupancy.rowCount(); i++) {
            String getKey = dataWithOccupancy.getString(i, "periodStart");
            dataWithOccupancy.row(i).setInt("occupancySum", occupancySumDataMap.get(getKey));
        }

        dataWithOccupancy.removeColumns("occupancySeconds");
        dataWithOccupancy = dataWithOccupancy.dropDuplicateRows();

        dataWithOccupancy.replaceColumn("occupancySum", // seconds to percents
                dataWithOccupancy.intColumn("occupancySum").divide(sensorCount * (periodMinutes * 60) / 100)
                        .multiply(10).roundInt().divide(10)); // round .1 operation
        dataWithOccupancy.column(dataWithOccupancy.columnCount() - 1).setName("occupancyPercent");

        System.out.println("Data is processed. Data without weather " + dataWithOccupancy.shape());
        Table dataWithOccupancyAndWeather = addingWetter(dataWithOccupancy);

        // adding attributes of the base of processed table
        dataWithOccupancyAndWeather.addColumns(IntColumn.create("weekDay", dataWithOccupancyAndWeather.rowCount()),
                IntColumn.create("month", dataWithOccupancyAndWeather.rowCount()),
                IntColumn.create("year", dataWithOccupancyAndWeather.rowCount()),
                IntColumn.create("timeSlot", dataWithOccupancyAndWeather.rowCount()), // timeslot in day
                DoubleColumn.create("previousOccupancy", dataWithOccupancyAndWeather.rowCount()), // occupancy N minutes ago
                DoubleColumn.create("occupancy", dataWithOccupancyAndWeather.rowCount())); // occupancy now

        // if horizon is less than an hour, hour
        int periodsInHour = 1;
        if (periodMinutes < 60)
            periodsInHour = 60 / periodMinutes;

        for (int i = 0; i < dataWithOccupancyAndWeather.rowCount(); i++) {
            LocalDateTime tmpDate = LocalDateTime.ofInstant(Instant.ofEpochSecond(dataWithOccupancyAndWeather
                            .row(i).getLong("periodStartSeconds")),
                    TimeZone.getDefault().toZoneId());
            dataWithOccupancyAndWeather.row(i).setInt("weekDay", tmpDate.getDayOfWeek().getValue());
            dataWithOccupancyAndWeather.row(i).setInt("month", tmpDate.getMonthValue());
            dataWithOccupancyAndWeather.row(i).setInt("year", tmpDate.getYear());
            dataWithOccupancyAndWeather.row(i).setInt("timeSlot",
                    (tmpDate.getMinute() + tmpDate.getHour() * 60) / (60 / periodsInHour)); //timeslot in a day


            double previousOccupancy = 0;
            if (i > 0) {
                previousOccupancy = dataWithOccupancyAndWeather.row(i - 1).getDouble("occupancyPercent");
            } else {
                previousOccupancy = dataWithOccupancyAndWeather.row(i).getDouble("occupancyPercent");
            }
            dataWithOccupancyAndWeather.row(i).setDouble("previousOccupancy", previousOccupancy);
            dataWithOccupancyAndWeather.row(i).setDouble("occupancy",
                    dataWithOccupancyAndWeather.row(i).getDouble("occupancyPercent"));
        }

//        System.out.println(dataWithOccupancyAndWeather.first(10));
//        System.out.println(dataWithOccupancyAndWeather.rowCount());

        if (shift24h) {
            for (int i = 0; i < dataWithOccupancyAndWeather.rowCount() - (1440 / periodMinutes); i++) { //exclude the last 24h

                //Get occ value shifted for 24h
                double occ24 = dataWithOccupancyAndWeather.row(i + (1440 / periodMinutes)).getDouble("occupancy");
                dataWithOccupancyAndWeather.row(i).setDouble("occupancy", occ24); //Overwrite the occ column
            }

            int tablelength = dataWithOccupancyAndWeather.rowCount();
            dataWithOccupancyAndWeather = dataWithOccupancyAndWeather.dropRange(tablelength - (1440 / periodMinutes) + 1, tablelength); //drop the last 24h of data
            //NOTE: Previous Occupancy not updated, as value is not yet "seen" in use case of shifting
        }

        dataWithOccupancyAndWeather.removeColumns("periodStartSeconds", "occupancyPercent");

        System.out.println(dataWithOccupancyAndWeather.first(10));

        return dataWithOccupancyAndWeather;
    }

    /**
     * Return a Table with rows that refer to the time gap between currentDate and currentDate + duration
     * Additional columns: currentDate, currentDate + duration and occupation time in seconds (maximum 60 * duration)
     *
     * @param currentDate    The time by which filtering is performed and the occupation time is calculated
     * @param unfilteredData The data to be processed
     * @return filtered Table with time range and occupancy in this range
     */
    private static Table filteredByExactPeriod(LocalDateTime currentDate, Table unfilteredData, int periodDuration) {
        ZoneId zoneId = ZoneId.of("Europe/Paris"); // the timezone has to be defined
        // convert LocalDateTime to ZonedDateTime to extract seconds
        ZonedDateTime zonedPeriodStart = currentDate.atZone(zoneId),
                zonedPeriodEnd = (currentDate.plusMinutes(periodDuration)).atZone(zoneId);
        // Get the epoch second value from ZonedDateTime
        long periodStartSeconds = zonedPeriodStart.toEpochSecond(),
                periodEndSeconds = zonedPeriodEnd.toEpochSecond();

        // define arrival and departure seconds columns for better readability
        LongColumn arrivalSecondsColumn = unfilteredData.longColumn("arrival_unix_seconds").asLongColumn();
        LongColumn departureSecondsColumn = unfilteredData.longColumn("departure_unix_seconds").asLongColumn();

        // selectionBetween:        (start)arrival       departure(end)
        Selection selectionBetween = arrivalSecondsColumn.isGreaterThanOrEqualTo(periodStartSeconds)
                .and(departureSecondsColumn.isLessThan(periodEndSeconds)),                 //here is a bug
                // selectionBetween: arrival(start)                       (end)departure
                selectionOverlap = arrivalSecondsColumn.isLessThanOrEqualTo(periodStartSeconds)
                        .and(departureSecondsColumn.isGreaterThanOrEqualTo(periodEndSeconds)),
                // selectionBetween: arrival(start)        departure      (end)
                selectionArrivalBefore = arrivalSecondsColumn.isLessThanOrEqualTo(periodStartSeconds)
                        .and(departureSecondsColumn.isGreaterThanOrEqualTo(periodStartSeconds)),
                // selectionBetween:        (start)   arrival             (end)departure
                selectionDepartureLater = arrivalSecondsColumn.isLessThanOrEqualTo(periodEndSeconds)
                        .and(departureSecondsColumn.isGreaterThanOrEqualTo(periodEndSeconds));

        // filtering with for all 4 variants of intersection
        Table filteredData = unfilteredData.where(selectionBetween.or(selectionOverlap)
                .or(selectionArrivalBefore).or(selectionDepartureLater));

        // adding columns to concatenate with the main Table
        filteredData.addColumns(StringColumn.create("periodStart", filteredData.rowCount()),
                StringColumn.create("periodEnd", filteredData.rowCount()),
                LongColumn.create("occupancySeconds", filteredData.rowCount()),
                LongColumn.create("periodStartSeconds", filteredData.rowCount()));

        if (!filteredData.isEmpty()) {
            long stayStart = 0, stayFinish = 0;
            for (int i = 0; i < filteredData.rowCount(); i++) {
                // start time for duration seconds
                if (Integer.valueOf(filteredData.getString(i, "arrival_unix_seconds")) > periodStartSeconds)
                    stayStart = Integer.valueOf(filteredData.getString(i, "arrival_unix_seconds"));
                else stayStart = periodStartSeconds;
                // finish time for duration seconds
                if (Integer.valueOf(filteredData.getString(i, "departure_unix_seconds")) < periodEndSeconds)
                    stayFinish = Integer.valueOf(filteredData.getString(i, "departure_unix_seconds"));
                else stayFinish = periodEndSeconds;

                filteredData.row(i).setLong("occupancySeconds", stayFinish - stayStart); // occupancy duration
                filteredData.row(i).setString("periodStart", currentDate.toString());
                filteredData.row(i).setString("periodEnd", currentDate.plusMinutes(periodDuration).toString());
                filteredData.row(i).setLong("periodStartSeconds", periodStartSeconds);
            }
        }

        return filteredData;
    }

    /**
     * Adds forecast data to occupancy data
     *
     * @param parkingOccupacy Preprocessed occupancy table
     * @return Table object
     */
    private Table addingWetter(Table parkingOccupacy) throws SQLException, Exception {
        // weather from DB
        String pattern = "dd.MM.yyyy HH:mm:ss"; // pattern to format the date from string
        String query = "SELECT * FROM public.\"60_Minutes_Dataset_Air_Temperature_and_Humidity_" + settings.parkingId + "\";";
        ResultSet resultSetForWeather = conn.createStatement().executeQuery(query);
        Table weather = new DataFrameReader(new ReaderRegistry()).db(resultSetForWeather);
        weather.column("MESS_DATUM").setName("Date"); // change column data name
        weather.column("TT_TU").setName("Temp");
        weather.column("RF_TU").setName("Humidity");
        weather.removeColumns("STATIONS_ID", "QN_9", "eor"); // removing unnecessary data
        resultSetForWeather.getStatement().close();

        // Define the input and output date formats
        DateTimeFormatter inputFormatter = DateTimeFormatter.ofPattern("yyyyMMddHH"), // how date data saved
                // by import from csv pattern is "yyyy-MM-dd'T'HH:mm:ss.SSS"
                outputFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm"); // into what to format

        // Use the map() method to apply the date conversion to each value in the column
        StringColumn formattedDateColumn = StringColumn.create("periodStart");

        // iteration over the rows: convert and format the dates, and add them to the new column periodStart
        for (int i = 0; i < weather.rowCount(); i++) {
            LocalDateTime date = LocalDateTime.parse(weather.getString(i, "Date"), inputFormatter);
            String formattedDate = date.format(outputFormatter);
            formattedDateColumn.append(formattedDate);
        }
        weather.replaceColumn("Date", formattedDateColumn);

        // copy of data to process in future
        Table weatherInPeriods = weather.emptyCopy();

        if (periodMinutes > 60) {
            LocalDateTime START_DATE = LocalDateTime.parse(parkingOccupacy.getString(0, "periodStart"));
            // START_DATE rounded to hours
            START_DATE = START_DATE.truncatedTo(ChronoUnit.HOURS);
            for (Row rowInWeather : weather) {
                LocalDateTime tmpDate = LocalDateTime.parse(rowInWeather.getString("periodStart"));
                if (tmpDate.getHour() != START_DATE.getHour()) //if this hour is not start hour in occupancy table
                    continue;

                Table tmpOneRowTable = Table.create(); // tmp table to save data
                tmpOneRowTable.addColumns(StringColumn.create("periodStart", 1),
                        DoubleColumn.create("Temp", 1),
                        DoubleColumn.create("Humidity", 1));

                tmpOneRowTable.row(0).setString("periodStart", tmpDate.toString());
                tmpOneRowTable.row(0).setDouble("Temp", rowInWeather.getDouble("Temp"));
                tmpOneRowTable.row(0).setDouble("Humidity", rowInWeather.getDouble("Humidity"));
                weatherInPeriods.append(tmpOneRowTable);
                START_DATE = START_DATE.plusMinutes(periodMinutes);
            }
        } else {
            // from every hour (=row) in weather extract data and add to new row(s) in weatherInPeriods
            // as example period duration = 30 minutes; then, pro 1 row in weather 2 rows in weatherInPeriods
            for (Row rowInWeather : weather) {
                LocalDateTime tmpStart = LocalDateTime.parse(rowInWeather.getString("periodStart"));
                for (int i = 0; i < 60 / periodMinutes; i++) { // for every period which contains in 1 hour
                    Table tmpOneRowTable = Table.create(); // tmp table to save data
                    tmpOneRowTable.addColumns(StringColumn.create("periodStart", 1),
                            DoubleColumn.create("Temp", 1),
                            DoubleColumn.create("Humidity", 1));

                    tmpOneRowTable.row(0).setString("periodStart", tmpStart.plusMinutes(i * periodMinutes).toString());
                    tmpOneRowTable.row(0).setDouble("Temp", rowInWeather.getDouble("Temp"));
                    tmpOneRowTable.row(0).setDouble("Humidity", rowInWeather.getDouble("Humidity"));
                    weatherInPeriods.append(tmpOneRowTable);
                }
            }
        }

        // convert periodStart to String (initially LocalDate)
        parkingOccupacy.addColumns(StringColumn.create("periodStartString", parkingOccupacy.rowCount()));
        int columnIndex = parkingOccupacy.columnIndex("periodStart");
        for (int i = 0; i < parkingOccupacy.rowCount(); i++) {
            parkingOccupacy.row(i).setString("periodStartString", parkingOccupacy.getString(i, columnIndex));
        }
        parkingOccupacy.removeColumns("periodStart");
        parkingOccupacy.column("periodStartString").setName("periodStart");

        // join tables based on "periodStart" column
        Table parkingOccupancyWithWetter = weatherInPeriods.joinOn("periodStart").inner(parkingOccupacy);
        parkingOccupancyWithWetter.removeColumns("periodEnd", "periodStart");

        System.out.println("Tables joined, shape " + parkingOccupancyWithWetter.shape());
        return parkingOccupancyWithWetter;
    }

    private String generateModelName(String att, String clas, int perMin, int td_size, boolean shift24h) {
        String shift = null;
        String cleanedAtt = att.replace(" ", "");
        String shortClas = "";
        switch (clas) {
            case "0":
                shortClas = "dt";
                break;
            case "1":
                shortClas = "rf";
                break;
            case "2":
                shortClas = "lr";
                break;
            case "3":
                shortClas = "knn";
                break;
        }

        if (shift24h) {
            shift = "-24h";
            return shortClas + "-" + cleanedAtt + "-" + perMin + "-" + td_size + shift;
        }
        return shortClas + "-" + cleanedAtt + "-" + perMin + "-" + td_size;
    }

    private Properties changeValues(String settingsPath, int pID, String att, String clas, int perMin, int td_size, boolean shift24h) {
        try {
            att = att.substring(0, att.length() - 2); //remove last ", "

            FileInputStream in = new FileInputStream("src/" + settingsPath);
            Properties props = new Properties();
            props.load(in);
            in.close();

            FileOutputStream out = new FileOutputStream("src/" + settingsPath);
            props.setProperty("parkingId", String.valueOf(pID));
            props.setProperty("attributes", "{" + att + "}");
            props.setProperty("classifiers", "{" + clas + "}");
            props.setProperty("periodMinutes", String.valueOf(perMin));
            props.setProperty("trainingDataSize", String.valueOf(td_size));
            props.setProperty("modelName", generateModelName(att, clas, perMin, td_size, shift24h));

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
            String settingsPath = "main/java/pipeline.properties";
            InputStream input = ModelTrainer.class.getClassLoader().getResourceAsStream(settingsPath);
            Properties props = new Properties();
            props.load(input);
            Settings settings = new Settings(settingsPath, props);
            ModelTrainer trainer = new ModelTrainer(settings);
            trainer.createDBConnection();

            String clas_val;
            String att_val;
            int pID_val;
            int perMin_val;
            int tdSize_val;
            boolean shift24h = false;

            //Parking Lot
            for (int pID = 0; pID <= 1; pID++) {
                pID_val = trainer.parkingLotMap.get(pID);

                //Period Minutes
                for (int perMin = 0; perMin <= 3; perMin++) {
                    perMin_val = trainer.periodMinuteMap.get(perMin).get(0);

                    //set flag for 24h occupancy prediction used in preprocessing
                    if (perMin == 3) {
                        shift24h = true;
                    }

                    //Training Data Size
                    for (int tdSize = 1; tdSize <= 2; tdSize++) {
                        tdSize_val = trainer.periodMinuteMap.get(perMin).get(tdSize);

                        //Classifier
                        for (int clas = 0; clas <= 3; clas++) {
                            clas_val = String.valueOf(clas);

                            //Attributes
                            for (int att0 = 0; att0 <= 1; att0++) {
                                for (int att1 = 0; att1 <= 1; att1++) {
                                    for (int att2 = 0; att2 <= 1; att2++) {
                                        for (int att3 = 0; att3 <= 1; att3++) {
                                            for (int att4 = 0; att4 <= 1; att4++) {
                                                for (int att5 = 0; att5 <= 1; att5++) {
                                                    for (int att6 = 0; att6 <= 1; att6++) {
                                                        att_val = "";
                                                        if (att0 + att1 + att2 + att3 + att4 + att5 + att6 == 0) {
                                                            // all attributes 0 = same as all 1
                                                            continue;
                                                        }
                                                        if (att0 == 1) {
                                                            att_val = att_val + "0, ";
                                                        }
                                                        if (att1 == 1) {
                                                            att_val = att_val + "1, ";
                                                        }
                                                        if (att2 == 1) {
                                                            att_val = att_val + "2, ";
                                                        }
                                                        if (att3 == 1) {
                                                            att_val = att_val + "3, ";
                                                        }
                                                        if (att4 == 1) {
                                                            att_val = att_val + "4, ";
                                                        }
                                                        if (att5 == 1) {
                                                            att_val = att_val + "5, ";
                                                        }
                                                        if (att6 == 1) {
                                                            att_val = att_val + "6, ";
                                                        }

                                                        //Initialize new Object for every iteration
                                                        props = trainer.changeValues(settingsPath, pID_val, att_val,
                                                                clas_val, perMin_val, tdSize_val, shift24h);
                                                        settings = new Settings(settingsPath, props);
                                                        trainer = new ModelTrainer(settings);

                                                        ResultSet rs = trainer.queryDB();
                                                        Table tableData = trainer.preprocessing(rs, shift24h);
                                                        trainer.saveQueryAsInstances(tableData);
                                                        rs.getStatement().close(); // closes the resource

                                                        // classifiers building
                                                        if (settings.classifiersData.isEmpty()) {
                                                            System.err.println("Classifier cannot be empty!");
                                                            break;
                                                        } else {
                                                            for (int i : settings.classifiersData) {
                                                                trainer.classifierMap.get(i).buildClassifier
                                                                        (trainer.m_Train_Data);
                                                            }
                                                        }
                                                        trainer.testClassifier();
                                                        trainer.saveModelToDB();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
