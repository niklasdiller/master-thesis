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
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

import java.io.*;
import java.sql.*;
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

public class ModelTrainer implements Serializable {

  private ArrayList<String> occupancyPredictAttributes = new ArrayList<String>() {{
    add("Temp");
    add("Humidity");
    add("weekDay");
    add("periodStartSeconds");
    add("occupancyPercent");
  }};


  /** All settings specified in properties file */
  private final Settings settings;

  /** Parking ID*/
  private int parkingId;

  /** Database connection */
  private Connection conn;

  /** Period in minutes */
  private int periodMinutes;

  /** The training data gathered so far. */
  private Instances m_Train_Data;

  /** The testing data gathered so far. */
  private Instances m_Test_Data;

  /** Random Forest maximal depth */
  private int randomForstMaxDepth = (new Settings("main/java/config.properties")).randomForestMaxDepth;

  /** k-Nearest Neighbors number of neighbours */
  private int kNeighbours = (new Settings("main/java/config.properties")).kNeighbours;

  /** The Decision Tree classifier. */
  private M5P m_DecisionTreeClassifier = new M5P();
  /** Decision Tree accuracy variables */
  private int correctPredictedDT;
  private double MAE_DT;
  private double MSE_DT;

  /** The Random Forest classifier. */
  private RandomForest m_RandomForestClassifier = new RandomForest();
  /** Random Forest accuracy variables */
  private int correctPredictedRF;
  private double MAE_RF;
  private double MSE_RF;

  /** The Linear Regression classifiert **/
  private LinearRegression m_LinearRegressionClassifier = new LinearRegression();
  /** Linear Regression accuracy variables */
  private int correctPredictedLR;
  private double MAE_LR;
  private double MSE_LR;

  /** The k-Nearest Neighbors classifier. */
  private IBk m_KNNClassifier = new IBk();
  /** k-Nearest Neighbors accuracy variables */
  private int correctPredictedKNN;
  private double MAE_KNN;
  private double MSE_KNN;

  /** Map for classifier choice **/
  private Map<Integer, Classifier> classifierMap = new HashMap<Integer, Classifier>();

  /** Map for classifier names **/
  private Map<Integer, String> classifierNamesMap = new HashMap<Integer, String>();

  /** Map for classifier names **/
  private Map<Integer, String> attributesNamesMap = new HashMap<Integer, String>();

  /** The filter */
  private final StringToNominal m_Filter = new StringToNominal();


  /**
   * Create a model trainer
   * @param settings Contains all settings to run training pipeline
   */
  public ModelTrainer(Settings settings) throws IOException, ParseException {
    this.settings = settings;
    this.periodMinutes = settings.periodMinutes;
    this.parkingId = settings.parkingId;

    String nameOfDataset = "ParkingOccupancyProblem";

    ArrayList<Attribute> attributes = new ArrayList<>();
    if (settings.attributesData.isEmpty()) {
      for (int i = 0; i < occupancyPredictAttributes.size()-1; i++) {
        attributes.add(new Attribute(occupancyPredictAttributes.get(i)));
      }
    }
    else {
      for (int i : settings.attributesData) {
        attributes.add(new Attribute(occupancyPredictAttributes.get(i)));
      }
    }
    Attribute occupancyAttribute = new Attribute(occupancyPredictAttributes.get(occupancyPredictAttributes.size()-1));
    attributes.add(occupancyAttribute);

    int targetAttributIndex = attributes.indexOf(occupancyAttribute);

    // create dataset with initial capacity of 10000 (?)
    m_Train_Data = new Instances(nameOfDataset, attributes, 10000);
    // add label at index targetAttributIndex of output attributes
    m_Train_Data.setClassIndex(targetAttributIndex);

    m_Test_Data = new Instances(nameOfDataset, attributes, 10000);
    m_Test_Data.setClassIndex(targetAttributIndex);

    this.randomForstMaxDepth = settings.randomForestMaxDepth;
    this.kNeighbours = settings.kNeighbours;

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

    this.attributesNamesMap.put(0, "temperature");
    this.attributesNamesMap.put(1, "humidity");
    this.attributesNamesMap.put(2, "day of the week");
    this.attributesNamesMap.put(2, "timestamp");

    // hyperparameter
    this.m_RandomForestClassifier.setMaxDepth(settings.randomForestMaxDepth);
    this.m_KNNClassifier.setKNN(settings.kNeighbours);

  }

  private void handInputTest() throws Exception {
    Attribute attribute1 = new Attribute("periodStartSeconds");
    Attribute attribute2 = new Attribute("weekDay");
    Attribute classAttribute = new Attribute("occupancyPercent");


    // Create an empty Instances object with the defined attributes
    Instances dataset = new Instances("Instance", new ArrayList<>(Arrays.asList(attribute1, attribute2, classAttribute)), 0);
    dataset.setClassIndex(dataset.numAttributes() - 1);

    // Create the instance and set its values
    Instance instance = new DenseInstance(3);
    instance.setValue(attribute1, 1623628800);
    instance.setValue(attribute2, 1);

    // Add the instance to the dataset
    dataset.add(instance);

    Instance instance2 = new DenseInstance(3);
    instance.setValue(attribute1, 1623675600);
    instance.setValue(attribute2, 1);

    // Add the instance to the dataset
    dataset.add(instance);
    for (Instance ins : dataset) {
      System.out.println("Random Forest: " + this.m_RandomForestClassifier.classifyInstance(ins));
      System.out.println("Linear Regression: " + this.m_LinearRegressionClassifier.classifyInstance(ins));
      System.out.println("KNN: " + this.m_KNNClassifier.classifyInstance(ins));
    }

  }

  /**
   * Create a connection to the CattleDB
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
   * @return A ResultSet containing the matching rows
   */
  private ResultSet queryDB() throws SQLException {
    System.out.println("Querying preprocessed data...");
    System.out.println("Data from table " + settings.preprocessTable + parkingId );
    String query = "SELECT * FROM public.\"" + settings.preprocessTable + parkingId +
            "\" ORDER BY arrival_unix_seconds ASC LIMIT "+ settings.tableLength +";";
    Statement st = conn.createStatement();
    return st.executeQuery(query);
  }

  /**
   * Create a new instance based on features calculated from windowData
   * @param rowData Row input data
   * @return Instance with calculated features and label
   * @throws Exception
   */
  private Instance createInstance(Row rowData) throws Exception {
    Instance instance = new DenseInstance(this.m_Train_Data.numAttributes());
    instance.setDataset(this.m_Train_Data);

    for (int attribute : settings.attributesData) {
      if (attribute == 0) {
        instance.setValue(this.m_Train_Data.attribute(occupancyPredictAttributes.get(attribute)),
                rowData.getDouble(occupancyPredictAttributes.get(attribute)));
      }
      else if (attribute == 2) {
        instance.setValue(this.m_Train_Data.attribute(occupancyPredictAttributes.get(attribute)),
                rowData.getInt(occupancyPredictAttributes.get(attribute)));
      }
      else if (attribute == 3) {
        instance.setValue(this.m_Train_Data.attribute(occupancyPredictAttributes.get(attribute)),
                rowData.getLong(occupancyPredictAttributes.get(attribute)));
      }
    }
    instance.setValue(this.m_Train_Data.attribute(occupancyPredictAttributes.get(occupancyPredictAttributes.size()-1)),
            rowData.getDouble(occupancyPredictAttributes.get(occupancyPredictAttributes.size()-1)));

    return instance;
  }

  /**
   * Convert the DB result to instances
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

    /* to extract test data after train data, not to mix
    double validSizeTest = this.settings.trainProp * table.rowCount();
    while (currentRowIndex < table.rowCount()) {
      instance = createInstance(currentRow);

      if (currentRowIndex < validSizeTest) {
        m_Data.add(instance);
      } else
        m_Test_Data.add(instance);

      currentRowIndex++;
      currentRow = table.row(currentRowIndex);
    }*/

    System.out.println("Converted data to instances.");
  }

  /**
   * Save the built classifier as a model file
   * @throws IOException
   */
  private void saveModelAsFile() throws IOException {
    String fileNameRF = "./" + settings.modelName + ".model";
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileNameRF));
    oos.writeObject(this.m_RandomForestClassifier);
    oos.flush();
    oos.close();

    System.out.println("Models saved at location " + fileNameRF);
  }

  /**
   * Save the model and all its parameters to the DB
   * @throws IOException
   * @throws SQLException
   */
  //@SuppressWarnings("SqlResolve")
  private void saveModelToDB() throws IOException, SQLException {
    // preparing to save
    String slotIDsString = "", classifierNamesString = "", attributesString = "";

    if (settings.slotsIDData.isEmpty()) {
      slotIDsString += "all IDs";
    }
    else {
      for (int id : settings.slotsIDData) {
        slotIDsString += (id + " ");
      }
    }

    if (settings.classifiersData.isEmpty()) {
      classifierNamesString += "all";
    }
    else {
      for (int classifierNumber : settings.classifiersData) {
        classifierNamesString += (classifierNamesMap.get(classifierNumber) + " ");
      }
    }

    if (settings.attributesData.isEmpty()) {
      attributesString += "all";
    }
    else {
      for (int attributNumber : settings.attributesData) {
        attributesString += (classifierNamesMap.get(attributNumber) + " ");
      }
    }

    String accuracyInfoDTString = "no classifier", accuracyInfoRFString = "no classifier",
            accuracyInfoLRString = "no classifier", accuracyInfoKNNString = "no classifier";
    List<Integer> listForClassifierIndexes = new ArrayList<Integer>();
    if (settings.classifiersData.isEmpty()) {
      listForClassifierIndexes.add(0);
      listForClassifierIndexes.add(1);
      listForClassifierIndexes.add(2);
      listForClassifierIndexes.add(3);
    }
    else {
      for (int i = 0; i < settings.classifiersData.size(); i++) {
        listForClassifierIndexes.add(settings.classifiersData.get(i));
      }
    }
    for (int i = 0; i < listForClassifierIndexes.size(); i++) {
      if (listForClassifierIndexes.get(i) == 0) {
        accuracyInfoDTString = "Correctly predicted: "
                + (double)Math.round((correctPredictedDT / (double)m_Test_Data.size()) * 100 * 100)/100 + "%"
                + " MAE: " + (double)Math.round(MAE_DT / (double)m_Test_Data.size()* 100)/100
                + " MSE: " +  (double)Math.round(MSE_DT / (double)m_Test_Data.size()* 100)/100;
      }
      else if (listForClassifierIndexes.get(i) == 1) {
        accuracyInfoRFString = "Correctly predicted: "
                + (double)Math.round((correctPredictedRF / (double) m_Test_Data.size()) * 100 * 100)/100 + "%"
        + " MAE: " + (double)Math.round(MAE_RF / (double)m_Test_Data.size()* 100)/100
        + " MSE: " +  (double)Math.round(MSE_RF / (double)m_Test_Data.size()* 100)/100;
      }
      else if (listForClassifierIndexes.get(i) == 2) {
        accuracyInfoLRString = "Correctly predicted: "
                + (double)Math.round((correctPredictedLR / (double)m_Test_Data.size()) * 100 * 100)/100 + "%"
                + " MAE: " + (double)Math.round(MAE_LR / (double)m_Test_Data.size()* 100)/100
                + " MSE: " +  (double)Math.round(MSE_LR / (double)m_Test_Data.size()* 100)/100;
      }
      else if (listForClassifierIndexes.get(i) == 3) {
        accuracyInfoKNNString = "Correctly predicted: "
                + (double) Math.round((correctPredictedKNN / (double) m_Test_Data.size()) * 100 * 100)/100 + "%"
                + " MAE: " + (double) Math.round(MAE_KNN / (double)m_Test_Data.size()* 100)/100
                + " MSE: " +  (double)Math.round(MSE_KNN / (double)m_Test_Data.size()* 100)/100;
      }
    }

      System.out.println("Saving model to database...");
    PreparedStatement ps = conn.prepareStatement("" +
            "INSERT INTO alex_trained_models (" +
            "model_name, parking_id, table_length, period_minutes, slotsIDs," +
            "classifiers, attributes, trainingDataProportion," +
            "accuracyPercent, randomForestMaxDepth, kNeighbours, " +
            "accuracyDT, accuracyRF, accuracyLR, accuracyKNN, decision_tree," +
            "random_forest, linear_regression, k_nearest_neighbors) " +
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);");

    // model_name
    ps.setString(1, settings.modelName);
    // parking id
    ps.setInt(2, settings.parkingId);
    // length of table
    ps.setInt(3, settings.tableLength);
    // periods duration in minutes
    ps.setInt(4, settings.periodMinutes);
    // ids of parking slots to parse
    ps.setString(5, slotIDsString);
    // classifier names
    ps.setString(6, classifierNamesString);
    // attributes
    ps.setString(7, attributesString);
    // train percent
    ps.setDouble(8, settings.trainProp);
    // deviation percentage for accuracy calculation
    ps.setInt(9, settings.accuracyPercent);
    // max depth for Random Forest Classifier
    ps.setInt(10, settings.randomForestMaxDepth);
    // number of neighbours for k-Nearest Neighbours Classifier
    ps.setInt(11, settings.kNeighbours);

     // accuracy results
    ps.setString(12, accuracyInfoDTString);
    ps.setString(13, accuracyInfoRFString);
    ps.setString(14, accuracyInfoLRString);
    ps.setString(15, accuracyInfoKNNString);

    int parameterIndexToWrite = 16;
    for (int i = 0; i < 4; i++) {
      if (listForClassifierIndexes.isEmpty() || listForClassifierIndexes.contains(i)) {

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream out = new ObjectOutputStream(bos);
        if (i == 0) {
          out.writeObject(this.m_DecisionTreeClassifier);
        }
        else if (i == 1) {
          out.writeObject(this.m_RandomForestClassifier);
        }
        else if (i == 2) {
          out.writeObject(this.m_LinearRegressionClassifier);
        }
        else if (i == 3) {
          out.writeObject(this.m_KNNClassifier);
        }
        out.flush();
        byte[] serializedClassifier = bos.toByteArray();
        bos.close();
        ByteArrayInputStream bis = new ByteArrayInputStream(serializedClassifier);
        ps.setBinaryStream(parameterIndexToWrite+i, bis, serializedClassifier.length);
        bis.close();
      }
      else {
        ps.setString(parameterIndexToWrite+i, "no classifier");
      }
    }

    ps.executeUpdate();
    ps.close();

    System.out.println("Model saved in database.");
  }

  /**
   * Convert classifier object to base64 encoded string
   * @param classifierIndex index of classifier to convert
   * @return Classifier encoded in base64 string
   * @throws IOException
   */
  private String classifierToString(int classifierIndex) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ObjectOutputStream oos = new ObjectOutputStream(baos);
    if (classifierIndex == 0) {
      oos.writeObject(this.m_DecisionTreeClassifier);
    }
    else if (classifierIndex == 1) {
      oos.writeObject(this.m_RandomForestClassifier);
    }
    else if (classifierIndex == 2) {
      oos.writeObject(this.m_LinearRegressionClassifier);
    }
    else if (classifierIndex == 3) {
      oos.writeObject(this.m_KNNClassifier);
    }
    oos.close();
    return Base64.getEncoder().encodeToString(baos.toByteArray());
  }

  /**
   * Save encoded base64 string in file
   * @param base64 The content to be saved
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

   /**
   * Apply filter on saved instances
   * @throws Exception
   */
  private void applyFilter() throws Exception {
    System.out.println("Applying filter...");
    this.m_Filter.setAttributeRange("first");
    this.m_Filter.setInputFormat(this.m_Train_Data);
    this.m_Train_Data = Filter.useFilter(this.m_Train_Data, this.m_Filter);
    this.m_Train_Data = new Instances(this.m_Train_Data);
    this.m_Test_Data = Filter.useFilter(this.m_Test_Data, this.m_Filter);
    this.m_Test_Data = new Instances(this.m_Test_Data);
  }

  /**
   * Test classifier on all test instances
   * @throws Exception
   */
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
    }
    else {
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
        meanSqErr += difference*difference;
        if (prediction >= value - percentForAccuracy && prediction <= value + percentForAccuracy) {
          correctPredicted++;
        }
      }
      if (index == 0) {
        correctPredictedDT = correctPredicted;
        MAE_DT = meanAbsErr;
        MSE_DT = meanSqErr;
      }
      else if (index == 1) {
        correctPredictedRF = correctPredicted;
        MAE_RF = meanAbsErr;
        MSE_RF = meanSqErr;
      }
      else if (index == 2) {
        correctPredictedLR = correctPredicted;
        MAE_LR = meanAbsErr;
        MSE_LR = meanSqErr;
      }
      else if (index == 3) {
        correctPredictedKNN = correctPredicted;
        MAE_KNN = meanAbsErr;
        MSE_KNN = meanSqErr;
      }
      System.out.println("\nCorrectly predicted " + classifierNamesMap.get(index) + " "
              + (double) Math.round((correctPredicted / (double) m_Test_Data.size()) * 100 * 100)/100 + "%");
      System.out.println(classifierNamesMap.get(index) + " Mean Absolute Error: "
              + (double) Math.round(meanAbsErr / (double)m_Test_Data.size()* 100)/100);
      System.out.println(classifierNamesMap.get(index) + " Mean Squared Error: " +
              (double)Math.round(meanSqErr / (double)m_Test_Data.size()* 100)/100);
    }
  }

  /**
   * Return processed table of occupancy data
   * @param rs    Set of data to process
   * @throws Exception
   * @return Table object
   */
  private Table preprocessing(ResultSet rs) throws Exception {
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
    }
    else
      data = dataAllID.copy();

    data.setName("Parking data");
    System.out.println("Parking data is imported from DB, data shape is " + data.shape());
    // get number of sensors (unique values)
    int sensorCount = data.column("parking_space_id").countUnique();

    // delete unnecessary columns and sort asc
    data.removeColumns("parking_lot_id", "xml_id", "parking_space_id");
    data.sortAscendingOn("arrival_unix_seconds");

    // seconds to data convertation
    String pattern = "dd.MM.yyyy HH:mm:ss"; // pattern to format the date from string
    DateTimeFormatter formatter = DateTimeFormatter.ofPattern(pattern); // instance to format the date with pattern

    // getting the first arrival time in "arrival_local_time"
    String startDateString = data.getString(0, "arrival_local_time");
    LocalDateTime START_DATE = LocalDateTime.parse(startDateString, formatter);
    // START_DATE rounded to hours
    START_DATE = START_DATE.truncatedTo(ChronoUnit.HOURS); // Truncate to hours first // before java.time.temporal.ChronoUnit.HOURS
    if (periodMinutes < 60) {
      int remainder = START_DATE.getMinute() % periodMinutes;
      if (remainder != 0) {
        START_DATE = START_DATE.plusMinutes(periodMinutes - remainder); // Round up to the next 10-minute interval
      }
    }
    // getting the last arrival time in "arrival_local_time" and process as START_DATE
    String endDateString = data.getString(data.rowCount() - 1, "arrival_local_time");
    LocalDateTime END_DATE = LocalDateTime.parse(endDateString, formatter);
    END_DATE = END_DATE.truncatedTo(ChronoUnit.HOURS); // Truncate to hours first // before java.time.temporal.ChronoUnit.HOURS
    if (periodMinutes < 60) {
      int remainder = END_DATE.getMinute() % periodMinutes;
      if (remainder != 0) {
        END_DATE = END_DATE.plusMinutes(periodMinutes - remainder); // Round up to the next 10-minute interval
      }
    }

    // copy of data to process in future
    Table dataWithOccupacy = data.emptyCopy();
    // adding column to concatenate with new rows in future
    dataWithOccupacy.addColumns(StringColumn.create("periodStart", dataWithOccupacy.rowCount()),
            StringColumn.create("periodEnd", dataWithOccupacy.rowCount()),
            LongColumn.create("occupancySeconds", dataWithOccupacy.rowCount()),
            LongColumn.create("periodStartSeconds"));


    while (START_DATE.isBefore(END_DATE)) {
      dataWithOccupacy.append(filteredByExactPeriod(START_DATE, data, periodMinutes));
      START_DATE = START_DATE.plusMinutes(periodMinutes);
    }

    dataWithOccupacy.removeColumns("arrival_unix_seconds", "departure_unix_seconds",
            "arrival_local_time", "departure_local_time");  // removing unnecessary columns

    // adding day of the week parameter
    dataWithOccupacy.addColumns(IntColumn.create("weekDay", dataWithOccupacy.rowCount()));
    for (int i = 0; i < dataWithOccupacy.rowCount(); i++) {
      LocalDateTime tmpDate = LocalDateTime.parse(dataWithOccupacy.getString(i, "periodStart"));
      dataWithOccupacy.row(i).setInt("weekDay", tmpDate.getDayOfWeek().getValue());
    }

    // processing occupancy sum
    dataWithOccupacy.addColumns(IntColumn.create("occupancySum", dataWithOccupacy.rowCount()));
    Map<String, Integer> occupancySumDataMap = new HashMap<String, Integer>();

    // for every periodStart save the sum or add to sum of occupancy in HashMap
    for (int i = 0; i < dataWithOccupacy.rowCount(); i++) {
      String getKey = dataWithOccupacy.getString(i, "periodStart"); // the time and date
      int getValue = Integer.valueOf(dataWithOccupacy.getString(i, "occupancySeconds")); //occupancy
      if (occupancySumDataMap.containsKey(getKey))
        occupancySumDataMap.put(getKey, getValue + occupancySumDataMap.get(getKey));
      else
        occupancySumDataMap.put(getKey, getValue);
    }

    // and put the occupancy value in "occupancySum" column
    for (int i = 0; i < dataWithOccupacy.rowCount(); i++) {
      String getKey = dataWithOccupacy.getString(i, "periodStart");
      dataWithOccupacy.row(i).setInt("occupancySum", occupancySumDataMap.get(getKey));
    }

    dataWithOccupacy.removeColumns("occupancySeconds");
    dataWithOccupacy = dataWithOccupacy.dropDuplicateRows();

    dataWithOccupacy.replaceColumn("occupancySum", // seconds to percents
            dataWithOccupacy.intColumn("occupancySum").divide(sensorCount * (periodMinutes * 60) / 100)
                    .multiply(10).roundInt().divide(10)); // round .1 operation
    dataWithOccupacy.column(dataWithOccupacy.columnCount() - 1).setName("occupancyPercent");
    //dataWithOccupacy.write().csv("src/dataWithOccupacyTest.csv");

    System.out.println("Data is processed. Data without weather " + dataWithOccupacy.shape());
    return addingWetter(dataWithOccupacy);
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
   * @param parkingOccupacy Preprocessed occupancy table
   * @return Table object
   */
  private Table addingWetter(Table parkingOccupacy) throws SQLException, Exception {
      // weather from DB
      String pattern = "dd.MM.yyyy HH:mm:ss"; // pattern to format the date from string
      DateTimeFormatter formatter = DateTimeFormatter.ofPattern(pattern); // instance to format the date with pattern
      String query = "SELECT * FROM public.\"60_Minutes_Dataset_Air_Temperature_and_Humidity_"+ parkingId +"\";";
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
        START_DATE = START_DATE.truncatedTo(ChronoUnit.HOURS); // Truncate to hours first // before java.time.temporal.ChronoUnit.HOURS
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
      }
      else {
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

      // concatenate tables based on "periodStart" column
      Table parkingOccupancyWithWetter = weatherInPeriods.joinOn("periodStart").inner(parkingOccupacy);

    parkingOccupancyWithWetter.removeColumns("periodEnd", "periodStart");
      String allInstances38 = "src/parking38RowDataAllWithWeather.csv";
      String only5000Instances38 = "src/parking38RowData5000WithWeather.csv";

      System.out.println("Die letzten Zeilen: ");
      //parkingOccupancyWithWetter.write().csv(only5000Instances38);
      System.out.println("Tables joined, shape " + parkingOccupancyWithWetter.shape());
      return parkingOccupancyWithWetter;
  }

  public static void main(String[] args) {
    try {
      Settings settings = new Settings("main/java/config.properties"); // instead of config.properties - ich
      ModelTrainer trainer = new ModelTrainer(settings);

      trainer.createDBConnection();
      ResultSet rs = trainer.queryDB();

      Table tableData = trainer.preprocessing(rs);
      trainer.saveQueryAsInstances(tableData);
      rs.getStatement().close(); // closes the resource
      //trainer.applyFilter();

      if (settings.classifiersData.isEmpty()) {
        for (int i = 0; i < trainer.classifierMap.size(); i++) {
          trainer.classifierMap.get(i).buildClassifier(trainer.m_Train_Data);
        }
      }
      else {
        for (int i : settings.classifiersData) {
          trainer.classifierMap.get(i).buildClassifier(trainer.m_Train_Data);
        }
      }
      trainer.testClassifier();

      //trainer.handInputTest();

    trainer.saveModelToDB();

    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
