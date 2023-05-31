package main.java;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import tech.tablesaw.io.DataFrameReader;
import tech.tablesaw.io.ReaderRegistry;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
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

public class ModelTrainer {

  private ArrayList<String> occupancyPredictAttributes = new ArrayList<String>() {{
    add("Temp");
    add("Humidity");
    add("periodStartSeconds");
    add("occupancyPercent");
  }}; // TODO: add to other functions


  /** All settings specified in properties file */
  private final Settings settings;

  /** Parking ID*/
  private final int parkingId = 38;

  /** Database connection */
  private Connection conn;

  /** Period in minutes */
  private int periodMultiplier = 30;

  /** Array of parking slots to predict **/
  private int[] slotsIDs = new int[] {637, 600, 617};

  /** The training data gathered so far. */
  private Instances m_Data;

  /** The testing data gathered so far. */
  private Instances m_Test_Data;

  /** The Random Forest classifier. */
  private Classifier m_RandomForestClassifier = new RandomForest() {{setMaxDepth(500);}};

  /** The k-Nearest Neighbors classifier. */
  private Classifier m_KNN = new IBk();

  /** The Linear Regression classifiert **/
  private Classifier m_LinearRegressionClassifier = new LinearRegression();

  private MultiLayerNetwork m_LSTM;

  /** The filter */
  private final StringToNominal m_Filter = new StringToNominal();


  /**
   * Create a model trainer
   * @param settings Contains all settings to run training pipeline
   */
  public ModelTrainer(Settings settings) {
    this.settings = settings;

    String nameOfDataset = "ParkingOccupancyProblem";

    ArrayList<Attribute> attributes = new ArrayList<>();

    attributes.add(new Attribute(occupancyPredictAttributes.get(0)));
    attributes.add(new Attribute(occupancyPredictAttributes.get(1)));
    attributes.add(new Attribute(occupancyPredictAttributes.get(2)));
    Attribute occupancyAttribute = new Attribute(occupancyPredictAttributes.get(3));
    attributes.add(occupancyAttribute);

    int targetAttributIndex = attributes.indexOf(occupancyAttribute);

    // Create dataset with initial capacity of 100, and set index of class.
    m_Data = new Instances(nameOfDataset, attributes, 1000);
    // Add label at index 0 of output attributes
    m_Data.setClassIndex(targetAttributIndex);

    m_Test_Data = new Instances(nameOfDataset, attributes, 1000);
    m_Test_Data.setClassIndex(targetAttributIndex);
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
    System.out.println("Data from table " + settings.preprocessTable);
    String query = "SELECT * FROM public.\"" + settings.preprocessTable + parkingId + "\" ORDER BY arrival_unix_seconds ASC LIMIT 5000;";
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
    Instance instance = new DenseInstance(this.m_Data.numAttributes());
    instance.setDataset(this.m_Data);

    instance.setValue(this.m_Data.attribute(occupancyPredictAttributes.get(0)), rowData.getDouble(occupancyPredictAttributes.get(0)));
    instance.setValue(this.m_Data.attribute(occupancyPredictAttributes.get(1)), rowData.getDouble(occupancyPredictAttributes.get(1)));
    instance.setValue(this.m_Data.attribute(occupancyPredictAttributes.get(2)), rowData.getLong(occupancyPredictAttributes.get(2)));
    instance.setValue(this.m_Data.attribute(occupancyPredictAttributes.get(3)), rowData.getDouble(occupancyPredictAttributes.get(3)));

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
        m_Data.add(instance);
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

    System.out.println("Converted data to instances.");
  }

  /**
   * Save the built classifier as a model file
   * @throws IOException
   */
  private void saveModelAsFile() throws IOException {
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("./" + settings.modelName + "RF.model"));
    oos.writeObject(m_RandomForestClassifier);
    oos.flush();
    oos.close();
    oos = new ObjectOutputStream(new FileOutputStream("./" + settings.modelName + "LR.model"));
    oos.writeObject(m_LinearRegressionClassifier);
    oos.flush();
    oos.close();
    oos = new ObjectOutputStream(new FileOutputStream("./" + settings.modelName + "KNN.model"));
    oos.writeObject(m_KNN);
    oos.flush();
    oos.close();
    System.out.println("Models saved");
  }

  /**
   * Convert classifier object to base64 encoded string
   * @return Classifier encoded in base64 string
   * @throws IOException
   */
  private String classifierToString() throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ObjectOutputStream oos = new ObjectOutputStream(baos);
    oos.writeObject(this.m_RandomForestClassifier);
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
    this.m_Filter.setInputFormat(this.m_Data);
    this.m_Data = Filter.useFilter(this.m_Data, this.m_Filter);
    this.m_Data = new Instances(this.m_Data);
    this.m_Test_Data = Filter.useFilter(this.m_Test_Data, this.m_Filter);
    this.m_Test_Data = new Instances(this.m_Test_Data);
  }

  /**
   * Create a new LSTM model
   * @return LSTM (RNN) model
   */
  private MultiLayerNetwork RNNConfig() throws Exception {
    // a regression model, which can predict continuous values
    int featuresCount = m_Data.numAttributes() -1;
    MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(101)
            .updater(new Adam())
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(new LSTM.Builder().nIn(featuresCount).nOut(16).activation(Activation.TANH).build()) //LSTM(16), 1. lvl
            .layer(new LSTM.Builder().nIn(16).nOut(8).activation(Activation.TANH).build()) //LSTM(8), 2. level
            .layer(new DropoutLayer.Builder(0.2).build())
            .layer(new LSTM.Builder().nIn(8).nOut(4).activation(Activation.TANH).build()) //LSTM(4), 3. level
            .layer(4, new RnnOutputLayer.Builder().nIn(4).nOut(1) //output lvl, model.compile
                    .activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
            .build();

    MultiLayerNetwork model = new MultiLayerNetwork(config);
    return model;
  }

  /**
   * Return iterator object from train or test data set
   * @param choiceValue Train (0) or test (1) iterator to get
   * @throws Exception
   * @return iterator
   */
  private DataSetIterator getIterators(int choiceValue) throws Exception {
    int trainSize = m_Data.size(),
            testSize = m_Test_Data.size(),
            attributesSize = m_Data.numAttributes() - 1;

    double[][][] trainX = new double[trainSize][attributesSize][1];
    double[][] trainY = new double[trainSize][1];
    double[][][] testX = new double[testSize][attributesSize][1];
    double[][] testY = new double[testSize][1];


    for (int i = 0; i < trainSize; i++) {
      Instance instance = m_Data.get(i);
      trainY[i][0] = instance.classValue();
      for (int j = 0; j < attributesSize; j++) {
        trainX[i][j][0] = instance.value(j);
      }
    }

    for (int i = 0; i < testSize; i++) {
      Instance instance = m_Test_Data.get(i);
      testY[i][0] = instance.classValue();
      for (int j = 0; j < attributesSize; j++) {
        testX[i][j][0] = instance.value(j);
      }
    }

    INDArray inputArr = Nd4j.create(trainX);
    INDArray labelArr = Nd4j.create(trainY);
    INDArray labelArr3d = labelArr.reshape(labelArr.size(0), 1, 1);

    DataSet dataSetTrain = new DataSet(inputArr, labelArr3d);
    DataSetIterator trainIterator = new ListDataSetIterator<>(Collections.singletonList(dataSetTrain));

    INDArray inputArrTest = Nd4j.create(testX),
            labelArrTest = Nd4j.create(testY),
            labelArr3dTest = labelArrTest.reshape(labelArrTest.size(0), 1, 1);

    DataSet dataSetTest = new DataSet(inputArrTest, labelArr3dTest);
    DataSetIterator testIterator = new ListDataSetIterator<>(Collections.singletonList(dataSetTest));

    if (choiceValue == 0)
      return trainIterator;
    else
      return testIterator;
  }

  /**
   * Train and test LSTM model
   * @throws Exception
   */
  private void RNNTrainAndTest() {
    try {
      // a regression model, which can predict continuous values
      MultiLayerNetwork model = RNNConfig();
      model.init();

      DataSetIterator trainIterator = getIterators(0),
              testIterator = getIterators(1);

      model.fit(trainIterator); // may be doesn't work

      RegressionEvaluation testEvaluation = new RegressionEvaluation();
      while (testIterator.hasNext()) {
        DataSet dataSet = testIterator.next();
        INDArray features = dataSet.getFeatures();
        INDArray labels = dataSet.getLabels();
        INDArray predicted = model.output(features, false);
        testEvaluation.eval(labels, predicted);
      }
      //there is only one column in testEvaluation
      System.out.println("\nLSTM TestEvaluation meanAbsoluteError: " + testEvaluation.meanAbsoluteError(0));

    }
    catch (Exception e) {
      System.out.println("Error in testRNN");
      System.out.println(e);
    }
  }

  /**
   * Test classifier on all test instances
   * @throws Exception
   */
  private void testClassifier() throws Exception {
    System.out.println("Testing model...");
    int correctPredRF = 0, correctPredLR = 0, correctPredKNN = 0;
    for (Instance i : m_Test_Data) {
      double value = i.classValue();
      double predictionRF = m_RandomForestClassifier.classifyInstance(i),
              predictionLR = m_LinearRegressionClassifier.classifyInstance(i),
              predictionKNN = m_KNN.classifyInstance(i);
      if (predictionRF >= value - 5 && predictionRF <= value + 5) {
        correctPredRF++;
      }
      if (predictionLR >= value - 5 && predictionLR <= value + 5) {
        correctPredLR++;
      }
      if (predictionKNN >= value - 5 && predictionKNN <= value + 5) {
        correctPredKNN++;
      }
    }
    double correctRateRF = correctPredRF / (double) m_Test_Data.size(),
            correctRateLR = correctPredLR / (double) m_Test_Data.size(),
            correctRateKNN = correctPredKNN / (double) m_Test_Data.size();
    System.out.println("Correctly predicted Random Forest: " + correctRateRF * 100 + "%");
    System.out.println("Correctly predicted Linear Regression: " + correctRateLR * 100 + "%");
    System.out.println("Correctly predicted k-Nearest Neighbors: " + correctRateKNN * 100 + "%");
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
    if (slotsIDs.length > 0) {
      for (int id : slotsIDs) {
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
    int remainder = START_DATE.getMinute() % periodMultiplier;
    if (remainder != 0) {
      START_DATE = START_DATE.plusMinutes(periodMultiplier - remainder); // Round up to the next 10-minute interval
    }
    // getting the last arrival time in "arrival_local_time" and process as START_DATE
    String endDateString = data.getString(data.rowCount() - 1, "arrival_local_time");
    LocalDateTime END_DATE = LocalDateTime.parse(endDateString, formatter);
    END_DATE = END_DATE.truncatedTo(ChronoUnit.HOURS); // Truncate to hours first // before java.time.temporal.ChronoUnit.HOURS
    remainder = END_DATE.getMinute() % periodMultiplier;
    if (remainder != 0) {
      END_DATE = END_DATE.plusMinutes(periodMultiplier - remainder); // Round up to the next 10-minute interval
    }

    // copy of data to process in future
    Table dataWithOccupacy = data.emptyCopy();
    // adding column to concatenate with new rows in future
    dataWithOccupacy.addColumns(StringColumn.create("periodStart", dataWithOccupacy.rowCount()),
            StringColumn.create("periodEnd", dataWithOccupacy.rowCount()),
            LongColumn.create("occupancySeconds", dataWithOccupacy.rowCount()),
            LongColumn.create("periodStartSeconds"));

    // from start date to end date iterate for every period periodMultiplier
    LocalDateTime tmpDate = START_DATE;

    while (!tmpDate.equals(END_DATE)) {
      dataWithOccupacy.append(filteredByExactPeriod(tmpDate, data, periodMultiplier));
      tmpDate = tmpDate.plusMinutes(periodMultiplier);
    }

    dataWithOccupacy.removeColumns("arrival_unix_seconds", "departure_unix_seconds",
            "arrival_local_time", "departure_local_time");  // removing unnecessary columns


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
            dataWithOccupacy.intColumn("occupancySum").divide(sensorCount * (periodMultiplier * 60) / 100)
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
      // from every hour (=row) in weather extract data and add to new row(s) in weatherInPeriods
      // as example period duration = 30 minutes; then, pro 1 row in weather 2 rows in weatherInPeriods
      for (Row rowInWeather : weather) {
          LocalDateTime tmpStart = LocalDateTime.parse(rowInWeather.getString("periodStart"));
        for (int i = 0; i < 60/periodMultiplier; i++) { // for every period which contains in 1 hour
            Table tmpOneRowTable = Table.create(); // tmp table to save data
            tmpOneRowTable.addColumns(StringColumn.create("periodStart", 1),
                    DoubleColumn.create("Temp", 1),
                    DoubleColumn.create("Humidity", 1));

          tmpOneRowTable.row(0).setString("periodStart", tmpStart.plusMinutes(i*periodMultiplier).toString());
          tmpOneRowTable.row(0).setDouble("Temp", rowInWeather.getDouble("Temp"));
          tmpOneRowTable.row(0).setDouble("Humidity", rowInWeather.getDouble("Humidity"));
          weatherInPeriods.append(tmpOneRowTable);

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
      trainer.applyFilter();

      trainer.m_RandomForestClassifier.buildClassifier(trainer.m_Data);
      trainer.m_LinearRegressionClassifier.buildClassifier(trainer.m_Data);
      trainer.m_KNN.buildClassifier(trainer.m_Data);

      trainer.testClassifier();
      trainer.RNNTrainAndTest();

    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
