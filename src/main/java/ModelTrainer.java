package main.java;

import org.json.simple.JSONArray;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.DataFrameReader;
import tech.tablesaw.io.ReaderRegistry;
import tech.tablesaw.io.jdbc.SqlResultSetReader;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

import java.io.*;
import java.sql.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import tech.tablesaw.api.*;
import tech.tablesaw.io.csv.CsvReadOptions;
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
  }};


  /** All settings specified in properties file */
  private final Settings settings;

  /** Database connection */
  private Connection conn;

  /** The different labels used for classification */
  private List<String> labels = new ArrayList<>();
  /** The training data gathered so far. */
  private Instances m_Data;

  /** The testing data gathered so far. */
  private Instances m_Test_Data;

  /** The actual classifier. */
  private final Classifier m_Classifier = new RandomForest();

  /** The filter */
  private final StringToNominal m_Filter = new StringToNominal();

  /** The names of the model's attributes */
  private ArrayList<String> outputAttributes = new ArrayList<>();

  /** The model's accuracy determined by the test */
  private double testAccuracy;

  /**
   * Create a model trainer
   * @param settings Contains all settings to run training pipeline
   */
  public ModelTrainer(Settings settings) {
    this.settings = settings;

    String nameOfDataset = "ParkingOccupancyRegressionProblem";

    ArrayList<Attribute> attributes = new ArrayList<>();

    attributes.add(new Attribute(occupancyPredictAttributes.get(0)));
    attributes.add(new Attribute(occupancyPredictAttributes.get(1)));
    attributes.add(new Attribute(occupancyPredictAttributes.get(2)));
    Attribute occupancyAttribute = new Attribute(occupancyPredictAttributes.get(3));
    attributes.add(occupancyAttribute);

    int targetAttributIndex = attributes.indexOf(occupancyAttribute);

    // Create dataset with initial capacity of 100, and set index of class.
    m_Data = new Instances(nameOfDataset, attributes, 100);
    // Add label at index 0 of output attributes
    m_Data.setClassIndex(targetAttributIndex);

    m_Test_Data = new Instances(nameOfDataset, attributes, 100);
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
    //String query = "SELECT * FROM public.\"Input_datasets_parkinglot_38\" ORDER BY arrival_unix_seconds ASC;";
    //String query = "SELECT * FROM public.\"Input_datasets_parkinglot_38\" ORDER BY arrival_unix_seconds ASC LIMIT 5000;";
    //String query = "SELECT * FROM public.\"Input_datasets_parkinglot_634\" ORDER BY arrival_unix_seconds ASC;";
    String query = "SELECT * FROM public.\"Input_datasets_parkinglot_634\" ORDER BY arrival_unix_seconds ASC LIMIT 5000;";
    Statement st = conn.createStatement();
    return st.executeQuery(query);
  }
  /**
   * Find the label with the most occurrences in window
   * @param windowData The windowed input data
   * @return Label with most occurrences
   * @throws Exception
   */
  private String getLabelForWindow(ArrayList<HashMap<String,String>> windowData) throws Exception {
    String[] labels = new String[windowData.size()];
    String binaryLabel = settings.binaryLabel;
    for (int i = 0; i < labels.length; i++) {
      String label = windowData.get(i).get("label");
      if (binaryLabel != null && !Objects.equals(label, binaryLabel)) {
        labels[i] = "Non" + binaryLabel;
      } else {
        labels[i] = label;
      }
    }

    List<String> detectedLabels = Arrays.stream(labels).distinct().collect(Collectors.toList());
    for (String x : detectedLabels){
      if (!this.labels.contains(x))
        this.labels.add(x);
    }

    // find label with most occurrences
    return Stream.of(labels).collect(Collectors.groupingBy(s -> s, Collectors.counting()))
            .entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElseThrow(() -> new Exception("No class label present"));
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
   * Convert db result to hashmap
   * @param rs db result
   * @return
   * @throws SQLException
   */
  private HashMap<String, String> extractDBResult(Row rs) throws Exception {
    HashMap<String, String> map = new HashMap<>();
    long arrivalSeconds = rs.getLong("periodStartSeconds");
    double temperature = rs.getDouble("Temp"),
            humidity = rs.getDouble("Humidity"),
            occupancyPercent = rs.getDouble("occupancyPercent");

    map.put("periodStartSeconds", Long.toString(arrivalSeconds));
    map.put("temperature", String.valueOf(temperature));
    map.put("humidity", String.valueOf(humidity));
    map.put("occupancyPercent", String.valueOf(occupancyPercent));

    return map;
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
    String fileName = "./" + settings.modelName + ".model";
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName));
    oos.writeObject(m_Classifier);
    oos.flush();
    oos.close();
    System.out.println("Saved model at location: " + fileName);
  }

  /**
   * Convert classifier object to base64 encoded string
   * @return Classifier encoded in base64 string
   * @throws IOException
   */
  private String classifierToString() throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ObjectOutputStream oos = new ObjectOutputStream(baos);
    oos.writeObject(this.m_Classifier);
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
   * Test classifier on all test instances
   * @throws Exception
   */
  private void testClassifier() throws Exception {
    System.out.println("Testing model...");
    int correctPred = 0;
    for (Instance i : m_Test_Data) {
      double value = i.classValue();
      double prediction = m_Classifier.classifyInstance(i);
      if (prediction >= value - 5 && prediction <= value + 5) {
        correctPred++;
      }
    }
    double correctRate = correctPred / (double) m_Test_Data.size();
    System.out.println("Correctly predicted: " + correctRate * 100 + "%");
    testAccuracy = correctRate * 100;
  }

  private Table preprocessing(ResultSet rs) throws Exception {
    Table data = new DataFrameReader(new ReaderRegistry()).db(rs);

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
    START_DATE = START_DATE.truncatedTo(java.time.temporal.ChronoUnit.HOURS);

    // getting the last arrival time in "arrival_local_time" and process as START_DATE
    String endDateString = data.getString(data.rowCount() - 1, "arrival_local_time");
    LocalDateTime END_DATE = LocalDateTime.parse(endDateString, formatter);
    END_DATE = END_DATE.truncatedTo(java.time.temporal.ChronoUnit.HOURS);


    // copy of data to process in future
    Table dataWithOccupacy = data.emptyCopy();
    // adding column to concatenate with new rows in future
    dataWithOccupacy.addColumns(StringColumn.create("periodStart", dataWithOccupacy.rowCount()),
            StringColumn.create("periodEnd", dataWithOccupacy.rowCount()),
            LongColumn.create("occupancySeconds", dataWithOccupacy.rowCount()),
            LongColumn.create("periodStartSeconds"));

    // from start date to end date iterate for every hour
    LocalDateTime tmpDate = START_DATE;

    while (!tmpDate.equals(END_DATE)) {
      dataWithOccupacy.append(filterRecordsByHours(tmpDate, data));
      tmpDate = tmpDate.plusHours(1);
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
            dataWithOccupacy.intColumn("occupancySum").divide(sensorCount * 3600 / 100)
                    .multiply(10).roundInt().divide(10)); // round .1 operation
    dataWithOccupacy.column(dataWithOccupacy.columnCount() - 1).setName("occupancyPercent");


    System.out.println("Data is processed, data with weather: " + dataWithOccupacy.shape());
    return addingWetter(dataWithOccupacy);
  }

  /**
   * returns a Table with rows that refer to the time gap between currentDate and currentDate + 1 hour
   * Additional columns: currentDate, currentDate + 1 hour and occupation time in seconds (maximum 3600 (1 hour))
   *
   * @param currentDate    The time by which filtering is performed and the occupation time is calculated
   * @param unfilteredData The data to be processed
   * @return filtered Table with time range and occupancy in this range
   */

  private static Table filterRecordsByHours(LocalDateTime currentDate, Table unfilteredData) {
    ZoneId zoneId = ZoneId.of("Europe/Paris"); // the timezone has to be defined
    // convert LocalDateTime to ZonedDateTime to extract seconds
    ZonedDateTime zonedcurrentDate = currentDate.atZone(zoneId),
            zonedcurrentDatePlusHour = (currentDate.plusHours(1)).atZone(zoneId);
    // Get the epoch second value from ZonedDateTime
    long currentDateSeconds = zonedcurrentDate.toEpochSecond(),
            currentDatePlusHourSeconds = zonedcurrentDatePlusHour.toEpochSecond();

    // define arrival and departure seconds columns for better readability
    LongColumn arrivalSecondsColumn = unfilteredData.longColumn("arrival_unix_seconds").asLongColumn();
    LongColumn departureSecondsColumn = unfilteredData.longColumn("departure_unix_seconds").asLongColumn();

    // selectionBetween:        (hour)arrival       departure(hour +1)
    Selection selectionBetween = arrivalSecondsColumn.isGreaterThanOrEqualTo(currentDateSeconds)
            .and(departureSecondsColumn.isLessThan(currentDatePlusHourSeconds)),                 //here is a bug
            // selectionBetween: arrival(hour)                       (hour +1)departure
            selectionOverlap = arrivalSecondsColumn.isLessThanOrEqualTo(currentDateSeconds)
                    .and(departureSecondsColumn.isGreaterThanOrEqualTo(currentDatePlusHourSeconds)),
            // selectionBetween: arrival(hour)        departure      (hour +1)
            selectionArrivalBefore = arrivalSecondsColumn.isLessThanOrEqualTo(currentDateSeconds)
                    .and(departureSecondsColumn.isGreaterThanOrEqualTo(currentDateSeconds)),
            // selectionBetween:        (hour)   arrival             (hour +1)departure
            selectionDepartureLater = arrivalSecondsColumn.isLessThanOrEqualTo(currentDatePlusHourSeconds)
                    .and(departureSecondsColumn.isGreaterThanOrEqualTo(currentDatePlusHourSeconds));

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
        if (Integer.valueOf(filteredData.getString(i, "arrival_unix_seconds")) > currentDateSeconds)
          stayStart = Integer.valueOf(filteredData.getString(i, "arrival_unix_seconds"));
        else stayStart = currentDateSeconds;
        // finish time for duration seconds
        if (Integer.valueOf(filteredData.getString(i, "departure_unix_seconds")) < currentDatePlusHourSeconds)
          stayFinish = Integer.valueOf(filteredData.getString(i, "departure_unix_seconds"));
        else stayFinish = currentDatePlusHourSeconds;

        filteredData.row(i).setLong("occupancySeconds", stayFinish - stayStart); // occupancy duration
        filteredData.row(i).setString("periodStart", currentDate.toString());
        filteredData.row(i).setString("periodEnd", currentDate.plusHours(1).toString());
        filteredData.row(i).setLong("periodStartSeconds", currentDateSeconds);
      }
    }

    return filteredData;
  }

  /**
   * Adds forecast data to occupancy data
   * Returns nothing, saves result in file
   */
  private Table addingWetter(Table parkingOccupacy) throws SQLException, Exception {
      // weather from DB
      //String query = "SELECT * FROM public.\"60_Minutes_Dataset_Air_Temperature_and_Humidity_38\";";
      String query = "SELECT * FROM public.\"60_Minutes_Dataset_Air_Temperature_and_Humidity_634\";";
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

      // convert periodStart to String (initially LocalDate)
      parkingOccupacy.addColumns(StringColumn.create("periodStartString", parkingOccupacy.rowCount()));
      int columnIndex = parkingOccupacy.columnIndex("periodStart");
      for (int i = 0; i < parkingOccupacy.rowCount(); i++) {
        parkingOccupacy.row(i).setString("periodStartString", parkingOccupacy.getString(i, columnIndex));
      }
      parkingOccupacy.removeColumns("periodStart");
      parkingOccupacy.column("periodStartString").setName("periodStart");

      // concatenate tables based on "periodStart" column
      Table parkingOccupancyWithWetter = weather.joinOn("periodStart").inner(parkingOccupacy);

    parkingOccupancyWithWetter.removeColumns("periodEnd", "periodStart");
      String allInstances38 = "src/parking38RowDataAllWithWeather.csv";
      String only5000Instances38 = "src/parking38RowData5000WithWeather.csv";

      parkingOccupancyWithWetter.write().csv(allInstances38);
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

      trainer.m_Classifier.buildClassifier(trainer.m_Data);
      trainer.applyFilter();

      trainer.testClassifier();

    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
