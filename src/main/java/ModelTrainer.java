package main.java;

import org.json.simple.JSONArray;
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

public class ModelTrainer {

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

    String nameOfDataset = "CattleClassificationProblem";

    ArrayList<Attribute> attributes = new ArrayList<>();

    attributes.add(new Attribute("label", (ArrayList<String>) null));

    // Add label at index 0 of output attributes
    outputAttributes.add("label");

    settings.featureData.forEach((field, features) -> {
      features.forEach(feature -> {
        String outputAttribute = field + feature;
        attributes.add(new Attribute(outputAttribute));
        outputAttributes.add(outputAttribute);
      });
    });

    // Create dataset with initial capacity of 100, and set index of class.
    m_Data = new Instances(nameOfDataset, attributes, 100);
    m_Data.setClassIndex(0);

    m_Test_Data = new Instances(nameOfDataset, attributes, 100);
    m_Test_Data.setClassIndex(0);
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
   * Preprocessing queries to generate a training dataset in database
   * @throws SQLException
   */
  @SuppressWarnings("SqlResolve")
  private void preprocessInDB() throws SQLException {
    String cattleId = settings.cattleId;
    String round = settings.round;
    String sensorTable = "sensor_data_" + cattleId + "_" + round;
    String labelsTable = "labels_data_" + cattleId + "_" + round;
    String combinedTable = "combined_data_" + cattleId + "_" + round;

    System.out.println("Start preprocessing...");
    Statement st = conn.createStatement();
    st.execute("DROP TABLE IF EXISTS " + sensorTable + ", " + labelsTable + ", " + combinedTable + ";");
    System.out.println("(1/4) Cleaned up old tables");
    st.execute("SELECT * INTO " + sensorTable +
            " FROM prediction_framework_sensor_data_rawdb" +
            " WHERE cattle_id = '" + cattleId + "' AND round = '" + round + "';");
    System.out.println("(2/4) Extracted sensor data into new table " + sensorTable);
    st.execute("SELECT * INTO " + labelsTable +
            " FROM prediction_framework_label_data_rawdb AS a" +
            " WHERE ((a.label='Gehen') OR (a.label='Grasen') OR (a.label='Liegen') OR (a.label='Stehen'))" +
            " AND cattle_id = '" + cattleId + "' AND round = '" + round + "';");
    System.out.println("(3/4) Extracted label data into new table " + labelsTable);
    st.execute("SELECT a.cattle_id, b.timestamp, a.label, b.gx, b.gy, b.gz, b.ax, b.ay, b.az INTO " + combinedTable +
            " FROM " + labelsTable + " as a, " + sensorTable + " as b" +
            " WHERE ((a.begin_time_epoch<=b.timestamp) AND (a.end_time_epoch>=b.timestamp));");
    System.out.println("(4/4) Joined data into new table " + combinedTable);
    st.close();

    System.out.println("Preprocessing complete.");
  }

  /**
   * Retrieve preprocessed data from database
   * @return A ResultSet containing the matching rows
   */
  @SuppressWarnings("SqlResolve")
  private ResultSet queryDBAfterPreprocessing() throws SQLException {
    System.out.println("Querying preprocessed data...");
    String query = "SELECT * FROM " + "combined_data_" + settings.cattleId + "_" + settings.round;
    if (settings.startTimestamp != null) {
      query += " WHERE timestamp >= " + settings.startTimestamp;
      if (settings.endTimestamp != null) {
        query += " AND timestamp <= " + settings.endTimestamp;
      }
    } else if (settings.endTimestamp != null) {
      query += " WHERE timestamp <= " + settings.endTimestamp;
    }
    query += " ORDER BY timestamp;";

    Statement st = conn.createStatement();
    return st.executeQuery(query);
  }

  /**
   * Get preprocessed data from previously existing table in db
   * @return A ResultSet containing the matching rows
   */
  private ResultSet queryDB() throws SQLException {
    System.out.println("Querying preprocessed data...");
    String query = "SELECT * FROM " + settings.preprocessTable + " ORDER BY sdid LIMIT 100;"; // instead of timestamp

    Statement st = conn.createStatement();
    return st.executeQuery(query);
  }

  /**
   * Calculate features for a specific field
   * @param features List of string codes for features which are to be calculated
   * @param field The field for which the features are calculated for e.g. "ax", "gyrMag"
   * @param values The input values for the calculation
   * @return Map of calculation results
   * @throws Exception
   */
  private HashMap<String, Double> calcFeatures(ArrayList<String> features, String field, double[] values) throws Exception {
    FeatureCalculator calc = new FeatureCalculator();
    HashMap<String, Double> results = new HashMap<>();
    for (String feature : features) {
      switch (feature) {
        case "min":
          results.put(field + feature, calc.min(values));
          break;
        case "max":
          results.put(field + feature, calc.max(values));
          break;
        case "mean":
          results.put(field + feature, calc.mean(values));
          break;
        case "median":
          results.put(field + feature, calc.median(values));
          break;
        case "stdev":
          results.put(field + feature, calc.std(values));
          break;
        case "iqr":
          results.put(field + feature, calc.iqr(values));
          break;
        case "skew":
          results.put(field + feature, calc.skew(values));
          break;
        case "kurt":
          results.put(field + feature, calc.kurtosis(values));
          break;
        case "rms":
          results.put(field + feature, calc.rms(values));
          break;
        case "mcr":
          results.put(field + feature, calc.mcr(values));
          break;
        case "energy":
          results.put(field + feature, calc.energy(values));
          break;
        case "peakfreq":
          results.put(field + feature, calc.peakFreq(values));
          break;
        case "freqentrpy":
          results.put(field + feature, calc.frDmEntropy(values));
          break;
        default:
          throw new Exception("Unknown feature in features setting");
      }
    }
    return results;
  }

  /**
   * Extract data and get calculated features for field
   * @param field The field for which the features are calculated for e.g. "ax", "gyrMag"
   * @param features List of string codes for features which are to be calculated
   * @param windowData Input data
   * @return The calculated results
   * @throws Exception
   */
  private HashMap<String, Double> getResultsForField(String field,  ArrayList<String> features, ArrayList<HashMap<String,String>> windowData) throws Exception {
    FeatureCalculator calc = new FeatureCalculator();
    double[] magValues = new double[3];
    HashMap<String, String> dataset;

    double[] values = new double[windowData.size()];
    for (int i = 0; i < values.length; i++) {
      dataset = windowData.get(i);
      switch (field) {
        case "gyrmag":
          magValues[0] = Double.parseDouble(dataset.get("gx"));
          magValues[1] = Double.parseDouble(dataset.get("gy"));
          magValues[2] = Double.parseDouble(dataset.get("gz"));
          values[i] = calc.mag(magValues);
          break;
        case "accmag":
          magValues[0] = Double.parseDouble(dataset.get("ax"));
          magValues[1] = Double.parseDouble(dataset.get("ay"));
          magValues[2] = Double.parseDouble(dataset.get("az"));
          values[i] = calc.mag(magValues);
          break;
        default:
          values[i] = Double.parseDouble(dataset.get(field));
      }
    }

    return calcFeatures(features, field, values);
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
   * @param windowData Raw input data
   * @return Instance with calculated features and label
   * @throws Exception
   */
  private Instance createInstance(ArrayList<HashMap<String,String>> windowData) throws Exception {
    HashMap<String, Double> instanceData = new HashMap<>();
    for (Map.Entry<String, ArrayList<String>> entry : settings.featureData.entrySet()) {
      String field = entry.getKey();
      ArrayList<String> features = entry.getValue();

      instanceData.putAll(getResultsForField(field, features, windowData));
    }

    Instance instance = new DenseInstance(this.m_Data.numAttributes());
    instance.setDataset(this.m_Data);

    String label = getLabelForWindow(windowData);
    Attribute labelAttr = m_Data.attribute("label");
    instance.setValue(labelAttr, label);

    instanceData.forEach((attribute, value) -> {
      Attribute attr = m_Data.attribute(attribute);
      instance.setValue(attr, value);
    });

    return instance;
  }

  /**
   * Convert db result to hashmap
   * @param rs db result
   * @return
   * @throws SQLException
   */
  private HashMap<String, String> extractDBResult(ResultSet rs) throws SQLException {
    HashMap<String, String> map = new HashMap<>();
    map.put("timestamp", rs.getString("timestamp"));
    map.put("label", rs.getString("label"));
    map.put("gx", rs.getString("gx"));
    map.put("gy", rs.getString("gy"));
    map.put("gz", rs.getString("gz"));
    map.put("ax", rs.getString("ax"));
    map.put("ay", rs.getString("ay"));
    map.put("az", rs.getString("az"));
    return map;
  }

  /**
   * Convert the DB result to instances
   * @param rs DB result
   */
  private void saveQueryAsInstances(ResultSet rs) throws Exception {
    long currStamp;
    boolean useBuffer = false;
    Instance instance;

    if (settings.windowStride < settings.windowSize) {
      useBuffer = true;
    }
    HashMap<String, String> dataset;
    ArrayList<HashMap<String,String>> windowData = new ArrayList<>();
    ArrayList<HashMap<String,String>> buffer = new ArrayList<>();
    int i = 0;
    int validSize = (int) (settings.trainProp * 10);

    rs.next();
    long windowStart = Long.parseLong(rs.getString("timestamp")); // start time from table
    do {
      currStamp = Long.parseLong(rs.getString("timestamp"));
      if (currStamp > windowStart + settings.windowSize) {
        instance = createInstance(windowData);

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

        windowData.clear();
        windowStart += settings.windowStride;

        if (useBuffer) {
          final long finalWindowStart = windowStart;
          buffer.forEach(elem -> {
            long timestamp = Long.parseLong(elem.get("timestamp"));
            if (timestamp >= finalWindowStart) {
              windowData.add(elem);
            } else {
              buffer.remove(elem);
            }
          });
        }
      }

      dataset = extractDBResult(rs);
      windowData.add(dataset);

      if (useBuffer) {
        buffer.add(dataset);
      }
    } while (rs.next());

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
   * Save classifier encoded in base64 string
   * @throws IOException
   */
  private void saveModelAsBase64String() throws IOException {
    System.out.println("Converting to base64 String");
    String classifierBase64 = this.classifierToString();
    String fileName = "./" + settings.modelName + ".txt";
    saveStringAsFile(classifierBase64, fileName);
  }

  /**
   * Save the model and all its parameters to the DB
   * @throws IOException
   * @throws SQLException
   */
  @SuppressWarnings("SqlResolve")
  private void saveModelToDB() throws IOException, SQLException {
    System.out.println("Saving model to database...");
    PreparedStatement ps = conn.prepareStatement("" +
            "INSERT INTO paul_trained_models(" +
            "model_name, labels, window_size, window_stride, features, validation_method, test_accuracy," +
            "created_time, train_test_split, train_table, output_attributes, binary_model, sensor_system," +
            "window_type, train_dataset, test_dataset, model_content)" +
            "VALUES (?,?,?,?,to_json(?::json),?,?,?,?,?,?,?,?,?,?,?,?);");

    // model_name
    ps.setString(1, settings.modelName);
    // labels
    ps.setString(2, JSONArray.toJSONString(labels));
    // window_size
    ps.setInt(3, settings.windowSize);
    // window_stride
    ps.setInt(4, settings.windowStride);
    // features
    ps.setString(5, settings.featuresJSON);
    // validation_method
    ps.setString(6, "accuracy");
    // test_accuracy
    ps.setDouble(7, testAccuracy);
    // created_time
    ps.setTimestamp(8, new Timestamp(System.currentTimeMillis()));
    // train_test_split
    ps.setDouble(9, settings.trainProp);
    // train_table
    if (settings.preprocessTable != null) {
      ps.setString(10, settings.preprocessTable);
    } else {
      ps.setString(10, "combined_data_" + settings.cattleId + "_" + settings.round);
    }
    // output_attributes
    ps.setString(11, JSONArray.toJSONString(outputAttributes));
    // binary_model
    ps.setBoolean(12, settings.binaryLabel != null);
    // sensor_system
    ps.setInt(13, 1);
    // window_type
    if (settings.windowStride == settings.windowSize) {
      ps.setString(14, "Jumping");
    } else if (settings.windowStride < settings.windowSize) {
      ps.setString(14, "Sliding");
    } else {
      ps.setString(14, "Sampling");
    }
    // train_dataset
    ps.setInt(15, 2);
    // test_dataset
    ps.setInt(16, 1);

    // model_content (the binary classifier)
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    ObjectOutputStream out = new ObjectOutputStream(bos);
    out.writeObject(m_Classifier);
    out.flush();
    byte[] serializedClassifier = bos.toByteArray();
    bos.close();
    ByteArrayInputStream bis = new ByteArrayInputStream(serializedClassifier);
    ps.setBinaryStream(17, bis, serializedClassifier.length);

    ps.executeUpdate();
    bis.close();
    ps.close();
    System.out.println("Saved model to database.");
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
   * Save the built model to the specified location
   * @throws IOException
   * @throws SQLException
   */
  private void saveModel() throws IOException, SQLException {
    switch (settings.saveIn) {
      case "db":
        saveModelToDB();
        break;
      case "file":
        saveModelAsFile();
        break;
      case "base64":
        saveModelAsBase64String();
        break;
    }
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
      if (value == prediction) {
        correctPred++;
      }
    }
    double correctRate = correctPred / (double) m_Test_Data.size();
    System.out.println("Correctly predicted: " + correctRate * 100 + "%");
    testAccuracy = correctRate * 100;
  }

  public static void main(String[] args) {
    try {
      Settings settings = new Settings("main/java/config.properties"); // instead of config.properties - ich
      ModelTrainer trainer = new ModelTrainer(settings);

      trainer.createDBConnection();

      ResultSet rs;
      if (settings.preprocessTable != null) {
        // use preprocessed data
        rs = trainer.queryDB();
      } else {
        // do preprocessing first
        if (settings.cattleId == null) {
          throw new NullPointerException("cattleId required in current setting");
        }
        if (settings.round == null) {
          throw new NullPointerException("round required in current setting");
        }


        trainer.preprocessInDB();
        rs = trainer.queryDBAfterPreprocessing();
      }

     trainer.saveQueryAsInstances(rs); // make batches of data
      /* rs.getStatement().close(); // closes the resource

      trainer.applyFilter();

      System.out.println("Building classifier...");;
      trainer.m_Classifier.buildClassifier(trainer.m_Data);

      trainer.testClassifier();

      trainer.saveModel();*/
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
