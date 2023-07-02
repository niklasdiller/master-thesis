package main.java;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Contains all the configurations
 * Parsed from config.properties file
 */
class Settings {
  private final Properties properties;

  String dbUri;
  String dbUsername;
  String dbPassword;
  int parkingId;
  int periodMinutes;
  int tableLength;
  int randomForestMaxDepth;
  int kNeighbours;
  int accuracyPercent;
  int trainTestStrategy;
  String featuresJSON;
  String slotsIDJSON;
  String classifiersJSON;
  String attributesJSON;
  HashMap<String, ArrayList<String>> featureData;
  List<Integer> slotsIDData;
  List<Integer> classifiersData;
  List<Integer> attributesData;
  String preprocessTable;
  String cattleId;
  String round;
  String binaryLabel;
  double trainProp;
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
    slotsIDJSON = getSettingAsString("slotsIDs", false);
    classifiersJSON = getSettingAsString("classifiers", false);
    attributesJSON = getSettingAsString("attributes", false);
    slotsIDData = parseStringToIntList(slotsIDJSON);
    classifiersData = parseStringToIntList(classifiersJSON);
    attributesData = parseStringToIntList(attributesJSON);
    parkingId = getSettingAsInt("parkingId", false);
    periodMinutes = getSettingAsInt("periodMinutes", false);
    tableLength = getSettingAsInt("tableLength", false);
    saveIn = getSettingAsString("saveIn", false);
    modelName = getSettingAsString("modelName", false);
    trainTestStrategy = getSettingAsInt("trainTestStrategy", false);

    // optional parameters
    preprocessTable = getSettingAsString("preprocessTable", true);
    trainProp = getTrainProp();
    randomForestMaxDepth = getSettingAsInt("randomForestMaxDepth", true);
    accuracyPercent = getSettingAsInt("accuracyPercent", true);
    kNeighbours = getSettingAsInt("kNeighbours", true);
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

  /**
   * Parse String input into a Integer List
   * @param stringToParse input string from settings
   * @return an Integer List containing extracted numbers from string
   * @throws ParseException
   */
  private static List<Integer> parseStringToIntList(String stringToParse) throws ParseException {
    String cleanedString = stringToParse.replace("{", "").replace("}", "")
            .replace(" ", "").replace(",", " ");
    List<Integer> slotsIDList = new ArrayList<>();

    if (!cleanedString.isBlank()) {
      slotsIDList = Arrays.stream(cleanedString.split("\\s"))
              .map(Integer::parseInt)
              .collect(Collectors.toList());
    }
    return slotsIDList;
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
