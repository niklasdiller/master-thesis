package main.java;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * Contains all the configurations
 * Parsed from config.properties file
 */
class Settings {
  private final Properties properties;

  String dbUri;
  String dbUsername;
  String dbPassword;
  int windowSize;
  int windowStride;
  String featuresJSON;
  HashMap<String, ArrayList<String>> featureData;
  String preprocessTable;
  String cattleId;
  String round;
  String binaryLabel;
  double trainProp;
  String startTimestamp;
  String endTimestamp;
  String saveIn;
  String modelName;

  Settings(String propertiesFile) throws IOException, ParseException {
    InputStream input = ModelTrainer.class.getClassLoader().getResourceAsStream(propertiesFile);
    properties = new Properties();
    properties.load(input);

    // mandatory parameters
    dbUri = getSettingAsString("dbUri", false);
    dbUsername = getSettingAsString("dbUsername", false);
    dbPassword = getSettingAsString("dbPassword", false);
    featuresJSON = getSettingAsString("features", false);
    featureData = parseFeatureData(featuresJSON);
    windowSize = getSettingAsInt("windowSize", false);
    windowStride = getSettingAsInt("windowStride", false);
    saveIn = getSettingAsString("saveIn", false);
    modelName = getSettingAsString("modelName", false);

    // optional parameters
    preprocessTable = getSettingAsString("preprocessTable", true);
    cattleId = getSettingAsString("cattleId", true);
    round = getSettingAsString("round", true);
    binaryLabel = getSettingAsString("binaryLabel", true);
    trainProp = getTrainProp();
    startTimestamp = getSettingAsString("startTimestamp", true);
    endTimestamp = getSettingAsString("endTimestamp", true);
  }

  private String getSettingAsString(String property, boolean optional) {
    String value = properties.getProperty(property);
    if (value == null && !optional) {
      throw new NullPointerException("No " + property + " found in the properties file.");
    }
    return value;
  }

  private int getSettingAsInt(String property, boolean optional) {
    String value = getSettingAsString(property, optional);
    if (value != null) {
      return Integer.parseInt(value);
    } else return -1;
  }

  /**
   * Parse features input string into a java object
   * @param featuresJSON input string from settings
   * @return a Hashmap containing the feature which will be calculated
   * @throws ParseException
   */
  private static HashMap<String, ArrayList<String>> parseFeatureData(String featuresJSON) throws ParseException {
    featuresJSON = featuresJSON.toLowerCase();
    HashMap<String, ArrayList<String>> featureData = new HashMap<>();
    JSONParser parser = new JSONParser();
    JSONObject json = (JSONObject) parser.parse(featuresJSON);

    Set<?> keys = json.keySet();
    keys.forEach(key -> {
      ArrayList<String> features = new ArrayList<>();
      JSONArray featureArr = (JSONArray) json.get(key);
      Object[] featureObjs =  featureArr.toArray();
      Arrays.stream(featureObjs).iterator().forEachRemaining(x -> features.add(x.toString()));
      featureData.put(key.toString(), features);
    });

    return featureData;
  }

  private double getTrainProp() {
    String trainPropValue = getSettingAsString("trainingDataProportion", true);
    double trainProp = 1;
    if (trainPropValue != null) {
      trainProp = Double.parseDouble(trainPropValue);
      if (trainProp <= 0 || trainProp > 1) {
        trainProp = 1;
      }
    }
    return trainProp;
  }
}
