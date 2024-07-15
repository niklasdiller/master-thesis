package main.java;

import tech.tablesaw.io.DataFrameReader;
import tech.tablesaw.io.ReaderRegistry;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
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
     * list of all possible features
     */
    private ArrayList<String> occupancyPredictFeatures = new ArrayList<String>() {{
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
     * feature indexes for actual model
     */
    private ArrayList<Integer> featuresIndexes = new ArrayList<>();

    /**
     * Map for featue names
     **/
    private ArrayList<String> featuresNamesList = new ArrayList<>();

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
    public static Connection conn;

    /**
     * Window Size in minutes
     */
    private int windowSize;

    /**
     * Flag to use when feature scaling is activated
     */
    private boolean featureScaling;

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

    private int corrPred;
    private double MAE;
    private double MSE;
    private double RMSE;
    /**
     * Decision Tree accuracy variables
     */
    private int correctPredictedDT;
    private double MAE_DT;
    private double MSE_DT;
    private double RMSE_DT;

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
    private double RMSE_RF;

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
    private double RMSE_LR;

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
    private double RMSE_KNN;

    /**
     * Map for classifier choice
     **/
    private Map<Integer, Classifier> classifierMap = new HashMap<Integer, Classifier>();

    /**
     * Map for classifier names
     **/
    private Map<Integer, String> classifierNamesMap = new HashMap<Integer, String>();

    /**
     * Map for windowSize values
     **/
    public Map<Integer, List<Integer>> windowSizeMap = new HashMap<>();

    /**
     * Number of Rows to use for training. Determined using trainingWeeks in settings.
     **/
    private int trainingDataSize;

    /**
     * Month and year of first entry of data used to train a model
     **/
    private String startOfTrainingData;

    /**
     * Map for parkingLot values
     **/
    public Map<Integer, Integer> parkingLotMap = new HashMap<>();

    /**
     * Map for maxDepthMap values for random forest classifier
     **/
    public Map<Integer, Integer> maxDepthMap = new HashMap<>();

    /**
     * Map for k values for KNN classifier
     **/
    public Map<Integer, Integer> kMap = new HashMap<>();

    /**
     * Map for k values for prediction Horizon values
     **/
    public Map<Integer, Integer> predHorMap = new HashMap<>();


    /**
     * Create a model trainer
     *
     * @param settings Contains all settings to run training pipeline
     */
    public ModelTrainer(Settings settings) throws Exception {
        this.settings = settings;
        this.windowSize = settings.windowSize;

        String nameOfDataset = "ParkingOccupancyProblem";

        ArrayList<Attribute> features = new ArrayList<>();
        if (Objects.equals(settings.settingsType, "training")) {
            if (settings.featuresData.isEmpty() || settings.featuresData == null) {
                for (int i = 0; i < occupancyPredictFeatures.size() - 1; i++) {
                    features.add(new Attribute(occupancyPredictFeatures.get(i)));
                    this.featuresIndexes.add(i);
                }
            } else {
                for (int i : settings.featuresData) {
                    features.add(new Attribute(occupancyPredictFeatures.get(i)));
                    this.featuresIndexes.add(i);
                }
            }
        }
        Attribute occupancyAttribute = new Attribute(occupancyPredictFeatures.get(occupancyPredictFeatures.size() - 1));
        features.add(occupancyAttribute);

        int targetAttributeIndex = features.indexOf(occupancyAttribute);

        // create dataset with initial capacity of 10000
        m_Train_Data = new Instances(nameOfDataset, features, 10000);
        // add label at index targetAttributeIndex of output features
        m_Train_Data.setClassIndex(targetAttributeIndex);

        m_Test_Data = new Instances(nameOfDataset, features, 10000);
        m_Test_Data.setClassIndex(targetAttributeIndex);

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

        // fill an features map with feature names
        this.featuresNamesList.add("temperature");
        this.featuresNamesList.add("humidity");
        this.featuresNamesList.add("day of the week");
        this.featuresNamesList.add("month");
        this.featuresNamesList.add("year");
        this.featuresNamesList.add("time slot");
        this.featuresNamesList.add("previous occupancy");

        // fill a WindowSizeMap with values for the training pipeline and their corresponding trainingDataSize
        // first value: windowSize
        // second value: First value of weeks of Training data to use
        // third value: Second value of weeks of Training data to use
        List<Integer> values0 = new ArrayList<>();
        values0.add(10); //window size 10 minutes
        values0.add(1); // 10080 minutes in a week -> 1008 10min slots in a week
        values0.add(4); // 40320 minutes in 4 weeks -> 4032 10min slots in 4 weeks
        this.windowSizeMap.put(0, values0);
        List<Integer> values1 = new ArrayList<>();
        values1.add(30);
        values1.add(1);
        values1.add(4);
        this.windowSizeMap.put(1, values1);
        List<Integer> values2 = new ArrayList<>();
        values2.add(60);
        values2.add(1);
        values2.add(4);
        this.windowSizeMap.put(2, values2);
        List<Integer> values3 = new ArrayList<>(); //24h Shift
        values3.add(60);
        values3.add(1);
        values3.add(4);
        this.windowSizeMap.put(3, values3);
        List<Integer> values4 = new ArrayList<>(); // Feature Scaling: Standardized
        values4.add(60);
        values4.add(4);
        this.windowSizeMap.put(4, values4);
//        values.clear();

        // fill a parkingLot Map with the corresponding Parking Lot IDs
        this.parkingLotMap.put(0, 38);
        this.parkingLotMap.put(1, 634);

        // hyperparameter, for RF and KNN
        this.m_RandomForestClassifier.setMaxDepth(settings.randomForestMaxDepth);
        this.m_KNNClassifier.setKNN(settings.kNeighbours);

        // fill a hyperparameter Map with changing values for maxdepth of random forest classifier
        this.maxDepthMap.put(0, 1);
        this.maxDepthMap.put(1, 5);
        this.maxDepthMap.put(2, 20);

        // fill a hyperparameter Map with changing values for k of KNN classifier
        this.kMap.put(0, 3);
        this.kMap.put(1, 19);
        this.kMap.put(2, 33);

        // fill a prediction Horizon Map with values of minutes that should be predicted in the future
        this.predHorMap.put(0, 10);
        this.predHorMap.put(1, 30);
        this.predHorMap.put(2, 60);

    }

    /**
     * Create a connection to the database
     *
     * @throws SQLException
     */
    public void createDBConnection() throws SQLException {
        Properties props = new Properties();
        props.setProperty("user", settings.dbUsername);
        props.setProperty("password", settings.dbPassword);
        conn = DriverManager.getConnection(settings.dbUri, props);

        System.out.println("Database connection established.");
    }

    /**
     * Get preprocessed data from previously existing table in db
     *
     * @return A ResultSet containing the matching rowsprep
     */
    public ResultSet queryDB() throws SQLException { //query all Data to put into preproc table
        System.out.println("Querying preprocessed data...");
        System.out.println("Data from table " + settings.rawTable + settings.parkingId);
        String query = "SELECT * FROM public.\"" + settings.rawTable + settings.parkingId +
                "\" ORDER BY arrival_unix_seconds ASC;";
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

        for (int feature : this.featuresIndexes) {
            if (feature == 0 || feature == 1 || feature == 6) {
                instance.setValue(this.m_Train_Data.attribute(occupancyPredictFeatures.get(feature)),
                        rowData.getDouble(occupancyPredictFeatures.get(feature)));
            } else if (feature == 2 || feature == 3 || feature == 4 || feature == 5) {
                instance.setValue(this.m_Train_Data.attribute(occupancyPredictFeatures.get(feature)),
                        rowData.getInt(occupancyPredictFeatures.get(feature)));
            }
        }
        // target is proceed separately, because featuresData contains only features
        instance.setValue(this.m_Train_Data.attribute(occupancyPredictFeatures.get(occupancyPredictFeatures.size() - 1)),
                rowData.getDouble(occupancyPredictFeatures.get(occupancyPredictFeatures.size() - 1)));

        return instance;
    }

    /**
     * Get the preprocessed data for model training from DB
     *
     * @param settings for the model to be trained
     * @param shift24h Flag for 24h Shift
     */
    private Table getPreprocData(Settings settings, boolean shift24h) throws SQLException {
        System.out.println("Data from table " + settings.preprocessedTable + ".");
        System.out.println("Filter for pID " + settings.parkingId + ", window size " + settings.windowSize + ".");
        System.out.println("Training Data Size is " + settings.trainingWeeks + " week(s) long.");

        //rows that make up trainingWeeks weeks
        trainingDataSize = settings.trainingWeeks * 7 * (1440 / settings.windowSize);

        //To get enough data for covering the future predictions
        int trainingDataTotal = trainingDataSize + settings.predictionHorizon;

        //Statements includes ORDER BY on timestamp of period start, so results are always displayed in the right order
        String query = "SELECT temp, humidity, weekday, month, year, time_slot, previous_occupancy, occupancy " +
                "FROM public.\"" + settings.preprocessedTable + "\"  WHERE pid = '" + settings.parkingId +
                "' AND window_size = '" + settings.windowSize + "' AND shift24h = '" + shift24h +
                "' ORDER BY period_start_time asc " +
                "LIMIT " + trainingDataTotal + ";";

        //System.out.println(query);
        System.out.println("Getting training data for model " + settings.modelName + ".");

        Statement st = conn.createStatement();
        ResultSet rs = st.executeQuery(query);

        Table res = new DataFrameReader(new ReaderRegistry()).db(rs);
        res.column("time_slot").setName("timeSlot"); //rename
        res.column("previous_occupancy").setName("previousOccupancy"); //rename

        //System.out.println(res.first(10));
        //System.out.println(res.last(10));

        // Data handling for prediction Horizon
        // Prev. Occ set to current Occ; Occ set to the value in predHor.-Minutes; other values stay the same
        for (int i = 0; i < trainingDataSize; i++) {
            res.row(i).setDouble("previousOccupancy", res.row(i).getDouble("occupancy"));
            res.row(i).setDouble("occupancy", res.row(i
                    + (settings.predictionHorizon / settings.windowSize)).getDouble("occupancy"));
        }

        //Shortening the training dataset to its original size
        res.dropRange(-settings.predictionHorizon);

        //Set startOfTrainingData
        startOfTrainingData = res.row(0).getInt("month") + "/";
        startOfTrainingData += res.row(0).getInt("year");

        rs.getStatement().close();
        System.out.println(res.first(10));
        //System.out.println(res.last(15));

        return res;
    }

    /**
     * Convert the DB result to instances
     *
     * @param table from preprocessing
     * @param fs    Flag for feature scaling
     */
    private void saveQueryAsInstances(Table table, boolean fs) throws Exception {
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

        // Apply feature scaling if flag is set
        if (fs) featureScale();

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
        String spaceIDsString = "", classifierNamesString = "", featuresString = "";

        if (settings.spaceIDData.isEmpty()) {
            spaceIDsString += "all IDs";
        } else {
            for (int id : settings.spaceIDData) {
                spaceIDsString += (id + " ");
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

        if (settings.featuresData.isEmpty()) {
            System.err.println("Features cannot be empty!"); //empty case handled through all selected
        } else {
            for (int featureNumber : settings.featuresData) {
                featuresString += (featuresNamesList.get(featureNumber) + ", ");
            }
            featuresString = featuresString.substring(0, featuresString.length() - 2); //remove last ", "
        }

        // Set the performance metrics
        double modelAccuracy, modelMae, modelMse, modelRmse;
        modelAccuracy = (double) Math.round((corrPred / (double) m_Test_Data.size()) * 100 * 100) / 100;
        modelMae = (double) Math.round(MAE / (double) m_Test_Data.size() * 100) / 100;
        modelMse = (double) Math.round(MSE / (double) m_Test_Data.size() * 100) / 100;
        modelRmse = (double) Math.round(Math.sqrt(MSE / (double) m_Test_Data.size()) * 100) / 100;

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

        // Obsolete, as performance metrics are now atomic

       /* for (int i = 0; i < listForClassifierIndexes.size(); i++) {
            if (listForClassifierIndexes.get(i) == 0) {
                accuracyInfoDTString = "Correctly predicted: "
                        + (double) Math.round((correctPredictedDT / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%"
                        + " MAE: " + (double) Math.round(MAE_DT / (double) m_Test_Data.size() * 100) / 100
                        + " MSE: " + (double) Math.round(MSE_DT / (double) m_Test_Data.size() * 100) / 100
                        + " RMSE: " + (double) Math.round(Math.sqrt(MSE_DT / (double) m_Test_Data.size()) * 100) / 100;

            } else if (listForClassifierIndexes.get(i) == 1) {
                accuracyInfoRFString = "Correctly predicted: "
                        + (double) Math.round((correctPredictedRF / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%"
                        + " MAE: " + (double) Math.round(MAE_RF / (double) m_Test_Data.size() * 100) / 100
                        + " MSE: " + (double) Math.round(MSE_RF / (double) m_Test_Data.size() * 100) / 100
                        + " RMSE: " + (double) Math.round(Math.sqrt(MSE_RF / (double) m_Test_Data.size()) * 100) / 100;

            } else if (listForClassifierIndexes.get(i) == 2) {
                accuracyInfoLRString = "Correctly predicted: "
                        + (double) Math.round((correctPredictedLR / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%"
                        + " MAE: " + (double) Math.round(MAE_LR / (double) m_Test_Data.size() * 100) / 100
                        + " MSE: " + (double) Math.round(MSE_LR / (double) m_Test_Data.size() * 100) / 100
                        + " RMSE: " + (double) Math.round(Math.sqrt(MSE_LR / (double) m_Test_Data.size()) * 100) / 100;

            } else if (listForClassifierIndexes.get(i) == 3) {
                accuracyInfoKNNString = "Correctly predicted: "
                        + (double) Math.round((correctPredictedKNN / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%"
                        + " MAE: " + (double) Math.round(MAE_KNN / (double) m_Test_Data.size() * 100) / 100
                        + " MSE: " + (double) Math.round(MSE_KNN / (double) m_Test_Data.size() * 100) / 100
                        + " RMSE: " + (double) Math.round(Math.sqrt(MSE_KNN / (double) m_Test_Data.size()) * 100) / 100;
            }
        }*/

        // saving in database
        System.out.println("Saving model to database...");
        // columns from table to save
        PreparedStatement ps = conn.prepareStatement("" +
                "INSERT INTO " + settings.tableName + " (" +
                "model_name, developer, train_test_strategy, created_time, model_size_in_bytes," +
                "parking_id, training_data_size, window_size, space_ids," +
                "classifiers, features, trainingDataProportion," +
                "accuracyPercent, randomForestMaxDepth, kNeighbours, " +
                "accuracy, mae, mse, rmse," +
                "model_content, window_stride, training_weeks, " +
                "start_of_training_data, prediction_horizon) " +
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"); // number of ? has to be the same as the number of columns

        ps.setString(1, settings.modelName);
        ps.setString(2, settings.developer);
        ps.setString(3, settings.trainTestText);
        ps.setString(4, formattedInstant);
        // ModelSize: parameterIndex 5. See down below
        ps.setInt(6, settings.parkingId);
        ps.setInt(7, trainingDataSize);     //number of rows of preprocessed data to be used for training
        ps.setInt(8, settings.windowSize);     // periods duration in minutes
        ps.setString(9, spaceIDsString);     // ids of parking spaces to parse
        ps.setString(10, classifierNamesString);     // classifier names
        ps.setString(11, featuresString); // features
        ps.setDouble(12, settings.trainProp); // train part
        ps.setInt(13, settings.accuracyPercent);     // deviation percentage for accuracy calculation

        if (Objects.equals(classifierNamesString, "Random Forest")) {
            ps.setInt(14, settings.randomForestMaxDepth); // max depth for Random Forest Classifier
        } else {
            ps.setInt(14, -1); //not applicable to other classifiers
        }

        if (Objects.equals(classifierNamesString, "K-Nearest Neighbours")) {
            ps.setInt(15, settings.kNeighbours); // number of neighbours for k-Nearest Neighbours Classifier
        } else {
            ps.setInt(15, -1); //not applicable to other classifiers
        }

        // Performance metrics
        ps.setDouble(16, modelAccuracy);
        ps.setDouble(17, modelMae);
        ps.setDouble(18, modelMse);
        ps.setDouble(19, modelRmse);

        // writing trained classifiers binary data
        int clasindex = listForClassifierIndexes.get(0);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream out = new ObjectOutputStream(bos);

        if (clasindex == 0) {
            out.writeObject(this.m_DecisionTreeClassifier);
        } else if (clasindex == 1) {
            out.writeObject(this.m_RandomForestClassifier);
        } else if (clasindex == 2) {
            out.writeObject(this.m_LinearRegressionClassifier);
        } else if (clasindex == 3) {
            out.writeObject(this.m_KNNClassifier);
        } else throw new NullPointerException("Classifier index is invalid.");

        out.flush();
        byte[] serializedClassifier = bos.toByteArray();
        modelSize += serializedClassifier.length;
        bos.close();
        ByteArrayInputStream bis = new ByteArrayInputStream(serializedClassifier);
        ps.setBinaryStream(20, bis, serializedClassifier.length);
        bis.close();


        ps.setInt(5, modelSize);
        ps.setInt(21, settings.windowSize); //Window Stride always same as Window Size
        ps.setInt(22, settings.trainingWeeks); //Weeks worth of training data
        ps.setString(23, startOfTrainingData); //Month and Year of first entry of training data for model
        ps.setInt(24, settings.predictionHorizon); //Prediction Horizon

        ps.executeUpdate(); // execution
        ps.close();

        System.out.println("Model saved to database" + settings.tableName + ".");
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

            corrPred = correctPredicted;
            MAE = meanAbsErr;
            MSE = meanSqErr;
            RMSE = Math.sqrt(meanSqErr);

            System.out.println("\nCorrectly predicted " + classifierNamesMap.get(index) + " "
                    + (double) Math.round((correctPredicted / (double) m_Test_Data.size()) * 100 * 100) / 100 + "%");
            System.out.println(classifierNamesMap.get(index) + " Mean Absolute Error: "
                    + (double) Math.round(meanAbsErr / (double) m_Test_Data.size() * 100) / 100);
            System.out.println(classifierNamesMap.get(index) + " Mean Squared Error: " +
                    (double) Math.round(meanSqErr / (double) m_Test_Data.size() * 100) / 100);
            System.out.println(classifierNamesMap.get(index) + " Root Mean Squared Error: " +
                    (double) Math.round(Math.sqrt(meanSqErr / (double) m_Test_Data.size()) * 100) / 100);
        }
    }

    /**
     * Return processed table of occupancy data
     *
     * @param rs       Set of data to process
     * @param shift24h Flag for 24hShift
     * @return Table object
     * @throws Exception
     */
    public Table preprocessing(ResultSet rs, boolean shift24h) throws Exception {
        Table dataAllID = new DataFrameReader(new ReaderRegistry()).db(rs);
        Table data = dataAllID.emptyCopy();

        // if there are special spaces to process and predict (e.g. for disabled people)
        if (!this.settings.spaceIDData.isEmpty()) {
            System.out.println("ID of parking spaces to parse: ");
            for (int id : settings.spaceIDData) {
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
        if (windowSize < 60) {
            int remainder = START_DATE.getMinute() % windowSize;
            if (remainder != 0) {
                START_DATE = START_DATE.plusMinutes(windowSize - remainder); // Round up to the next N-minute interval
            }
        }
        // getting the last arrival time in "arrival_local_time" and process as START_DATE
        String endDateString = data.getString(data.rowCount() - 1, "arrival_local_time");
        LocalDateTime END_DATE = LocalDateTime.parse(endDateString, formatter);
        END_DATE = END_DATE.truncatedTo(ChronoUnit.HOURS);
        if (windowSize < 60) {
            int remainder = END_DATE.getMinute() % windowSize;
            if (remainder != 0) {
                END_DATE = END_DATE.plusMinutes(windowSize - remainder);
            }
        }

        Table dataWithOccupancy = data.emptyCopy();     // copy of data to filter
        // adding column to concatenate with new rows
        dataWithOccupancy.addColumns(StringColumn.create("periodStart", dataWithOccupancy.rowCount()),
                StringColumn.create("periodEnd", dataWithOccupancy.rowCount()),
                LongColumn.create("occupancySeconds", dataWithOccupancy.rowCount()),
                LongColumn.create("periodStartSeconds"));

        while (START_DATE.isBefore(END_DATE)) {
            dataWithOccupancy.append(filteredByExactPeriod(START_DATE, data, windowSize));
            START_DATE = START_DATE.plusMinutes(windowSize);
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
                dataWithOccupancy.intColumn("occupancySum").divide(sensorCount * (windowSize * 60) / 100)
                        .multiply(10).roundInt().divide(10)); // round .1 operation
        dataWithOccupancy.column(dataWithOccupancy.columnCount() - 1).setName("occupancyPercent");

        System.out.println("Data is processed. Data without weather " + dataWithOccupancy.shape());
        Table dataWithOccupancyAndWeather = addingWetter(dataWithOccupancy);

        // adding features of the base of processed table
        dataWithOccupancyAndWeather.addColumns(IntColumn.create("weekDay", dataWithOccupancyAndWeather.rowCount()),
                IntColumn.create("month", dataWithOccupancyAndWeather.rowCount()),
                IntColumn.create("year", dataWithOccupancyAndWeather.rowCount()),
                IntColumn.create("timeSlot", dataWithOccupancyAndWeather.rowCount()), // timeslot in day
                DoubleColumn.create("previousOccupancy", dataWithOccupancyAndWeather.rowCount()), // occupancy N minutes ago
                DoubleColumn.create("occupancy", dataWithOccupancyAndWeather.rowCount()), // occupancy now
                DateTimeColumn.create("periodStartTime", dataWithOccupancyAndWeather.rowCount())); // time of period start

        // if horizon is less than an hour, hour
        int periodsInHour = 1;
        if (windowSize < 60)
            periodsInHour = 60 / windowSize;

        for (int i = 0; i < dataWithOccupancyAndWeather.rowCount(); i++) {
            LocalDateTime tmpDate = LocalDateTime.ofInstant(Instant.ofEpochSecond(dataWithOccupancyAndWeather
                            .row(i).getLong("periodStartSeconds")),
                    TimeZone.getDefault().toZoneId());

            dataWithOccupancyAndWeather.row(i).setInt("weekDay", tmpDate.getDayOfWeek().getValue());
            dataWithOccupancyAndWeather.row(i).setInt("month", tmpDate.getMonthValue());
            dataWithOccupancyAndWeather.row(i).setInt("year", tmpDate.getYear());
            dataWithOccupancyAndWeather.row(i).setInt("timeSlot",
                    (tmpDate.getMinute() + tmpDate.getHour() * 60) / (60 / periodsInHour)); //timeslot in a day
            dataWithOccupancyAndWeather.row(i).setDateTime("periodStartTime", tmpDate);

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

//        //24h Shift: Prediction Horizon for 24h, set in the preprocessing phase.
//        // New Pred. Hor. are being set after preprocessing in getPreprocData().
//        if (shift24h) {
//            for (int i = 0; i < dataWithOccupancyAndWeather.rowCount(); i++) {
//                LocalDateTime tmpDate = LocalDateTime.ofInstant(Instant.ofEpochSecond(dataWithOccupancyAndWeather
//                                .row(i).getLong("periodStartSeconds")),
//                        TimeZone.getDefault().toZoneId());
//               LocalDateTime newDate = tmpDate.minusHours(24);
//
//                dataWithOccupancyAndWeather.row(i).setInt("weekDay", newDate.getDayOfWeek().getValue());
//                dataWithOccupancyAndWeather.row(i).setInt("month", newDate.getMonthValue());
//                dataWithOccupancyAndWeather.row(i).setInt("year", newDate.getYear());
//                //because prev. occ. makes no sense in this case:
////                dataWithOccupancyAndWeather.row(i).setDouble("previousOccupancy", -1);
//            }
//        }

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

        if (windowSize > 60) {
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
                START_DATE = START_DATE.plusMinutes(windowSize);
            }
        } else {
            // from every hour (=row) in weather extract data and add to new row(s) in weatherInPeriods
            // as example period duration = 30 minutes; then, pro 1 row in weather 2 rows in weatherInPeriods
            for (Row rowInWeather : weather) {
                LocalDateTime tmpStart = LocalDateTime.parse(rowInWeather.getString("periodStart"));
                for (int i = 0; i < 60 / windowSize; i++) { // for every period which contains in 1 hour
                    Table tmpOneRowTable = Table.create(); // tmp table to save data
                    tmpOneRowTable.addColumns(StringColumn.create("periodStart", 1),
                            DoubleColumn.create("Temp", 1),
                            DoubleColumn.create("Humidity", 1));

                    tmpOneRowTable.row(0).setString("periodStart", tmpStart.plusMinutes(i * windowSize).toString());
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


    /**
     * Applies feature scaling to train and test dataset
     */
    private void featureScale() throws Exception {
        Normalize filter = new Normalize();
        filter.setInputFormat(m_Train_Data);
        m_Train_Data = Filter.useFilter(m_Train_Data, filter);
        m_Test_Data = Filter.useFilter(m_Test_Data, filter);
    }

    private String generateModelName(int pID, String att, String clas, int winSize, int weeks, int predHor,
                                     boolean shift24h, boolean fs) {
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
            String shift = "-24h";
            return shortClas + "-" + cleanedAtt + "-" + winSize + "-" + weeks + "-" + pID + "-" + predHor + shift;
        } else if (fs) {
            String fscale = "-fs";
            return shortClas + "-" + cleanedAtt + "-" + winSize + "-" + weeks + "-" + pID + "-" + predHor + fscale;
        }
        return shortClas + "-" + cleanedAtt + "-" + winSize + "-" + weeks + "-" + pID + "-" + predHor;
    }

    private void changeHyperparameters(String settingsPath, String clas_val, int val) {
        try {
            FileInputStream in = new FileInputStream("src/" + settingsPath);
            Properties props = new Properties();
            props.load(in);
            in.close();

            FileOutputStream out = new FileOutputStream("src/" + settingsPath);

            //if random forest, set maxDepth
            if (Objects.equals(clas_val, "1")) props.setProperty("randomForestMaxDepth", String.valueOf(val));

            //if KNN, set k
            if (Objects.equals(clas_val, "3")) props.setProperty("kNeighbours", String.valueOf(val));

            props.store(out, null);
            out.close();

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private Properties changeValues(String settingsPath, int pID, String feat, String clas, int winSize,
                                    int weeks, boolean shift24h, boolean fs, int predHor) {
        try {
            feat = feat.substring(0, feat.length() - 2); //remove last ", "

            FileInputStream in = new FileInputStream("src/" + settingsPath);
            Properties props = new Properties();
            props.load(in);
            in.close();

            FileOutputStream out = new FileOutputStream("src/" + settingsPath);
            props.setProperty("parkingId", String.valueOf(pID));
            props.setProperty("features", "{" + feat + "}");
            props.setProperty("classifiers", "{" + clas + "}");
            props.setProperty("windowSize", String.valueOf(winSize));
            props.setProperty("trainingWeeks", String.valueOf(weeks));
            props.setProperty("modelName", generateModelName(pID, feat, clas, winSize, weeks, predHor, shift24h, fs));
            props.setProperty("predictionHorizon", String.valueOf(predHor));

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
            String settingsPath = "main/java/training.properties";
            InputStream input = ModelTrainer.class.getClassLoader().getResourceAsStream(settingsPath);
            Properties props = new Properties();
            props.load(input);
            Settings settings = new Settings(settingsPath, props);
            ModelTrainer trainer = new ModelTrainer(settings);
            trainer.createDBConnection();

            String clas_val;
            String att_val;
            int pID_val;
            int winSize_val;
            int weeks_val;
            boolean shift24h = false;
            boolean fs = false;
            int hyperMax = 0; //End condition for for-loop for hyperparameters RF and KNN
            int predHor_val;

            //Parking Lot
            for (int pID = 0; pID <= trainer.parkingLotMap.size() - 1; pID++) {
                pID_val = trainer.parkingLotMap.get(pID);

                //Window Size
                for (int winSize = 0; winSize <= trainer.windowSizeMap.size() - 1; winSize++) {
                    winSize_val = trainer.windowSizeMap.get(winSize).get(0);

                    //Prediction Horizon
                    for (int predHor = 0; predHor <= trainer.predHorMap.size() - 1; predHor++) {
                        predHor_val = trainer.predHorMap.get(predHor);

                        //Training Data Size in Weeks. Initial value 1, see hashmap
                        for (int weeks = 1; weeks <= trainer.windowSizeMap.get(winSize).size() - 1; weeks++) {
                            weeks_val = trainer.windowSizeMap.get(winSize).get(weeks);

                            //Classifier
                            for (int clas = 0; clas <= 3; clas++) {
                                clas_val = String.valueOf(clas);

                                //Features
                                for (int att0 = 0; att0 <= 1; att0++) { //Temp
                                    for (int att1 = 0; att1 <= 1; att1++) { // Humidity
                                        for (int att2 = 0; att2 <= 1; att2++) { //Weekday
                                            for (int att3 = 0; att3 <= 1; att3++) { //Month
                                                for (int att4 = 0; att4 <= 1; att4++) { //Year
                                                    for (int att6 = 0; att6 <= 1; att6++) { // Previous Occupancy

                                                        //Get size of map for RF / KNN for for-loop end condition
                                                        // Different hyperparameters can therefore be set arbitrarily
                                                        if (clas_val.equals("1")) { //RF classifier
                                                            hyperMax = trainer.maxDepthMap.size() - 1;
                                                        } else if (clas_val.equals("3")) { // KNN classifier
                                                            hyperMax = trainer.kMap.size() - 1;
                                                        }

                                                        //if no changes to the hyperparameters should be made,
                                                        // remove this for-loop
                                                        for (int h = 0; h <= hyperMax; h++) {

                                                            //set flag for 24h occupancy prediction used in preprocessing
                                                            if (winSize == 3) shift24h = true;

                                                            //set flag for usage of feature scaling in preprocessing
                                                            if (winSize == 4) fs = true;

                                                            att_val = "";
                                                            if (att0 == 1) att_val = att_val + "0, ";
                                                            if (att1 == 1) att_val = att_val + "1, ";
                                                            if (att2 == 1) att_val = att_val + "2, ";
                                                            if (att3 == 1) att_val = att_val + "3, ";
                                                            if (att4 == 1) att_val = att_val + "4, ";
                                                            att_val += "5, "; //TimeSlot always a feature

                                                            if (att6 == 1 && !shift24h) { //as no prev. occ. in 24h shift
                                                                att_val = att_val + "6, ";
                                                            } else if (att6 == 1 && shift24h) { //to avoid duplicates
                                                                shift24h = false;
                                                                fs = false;
                                                                continue;
                                                            }

                                                            if (clas_val.equals("1")) {
                                                                trainer.changeHyperparameters
                                                                        (settingsPath, clas_val, trainer.maxDepthMap.get(h));
                                                            } else if (clas_val.equals("3")) {
                                                                trainer.changeHyperparameters
                                                                        (settingsPath, clas_val, trainer.kMap.get(h));
                                                            }

                                                            //Initialize new Object for every iteration
                                                            props = trainer.changeValues(settingsPath, pID_val, att_val,
                                                                    clas_val, winSize_val, weeks_val, shift24h, fs, predHor_val);

                                                            settings = new Settings(settingsPath, props);
                                                            trainer = new ModelTrainer(settings);

                                                            Table tableData = trainer.getPreprocData(settings, shift24h);
                                                            trainer.saveQueryAsInstances(tableData, fs);

                                                            shift24h = false; //reset shift flag
                                                            fs = false; //reset feature scale flag

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

                                                            if (!clas_val.equals("1") && !clas_val.equals("3")) {
                                                                break;
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
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
