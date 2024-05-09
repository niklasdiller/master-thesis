package main.java;

import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Contains all the configurations
 * Parsed from properties file
 */
class Settings {
    public Properties properties;

    String dbUri;
    String dbUsername;
    String dbPassword;
    String developer;
    int parkingId;
    int periodMinutes;
    int trainingWeeks;
    int randomForestMaxDepth;
    int kNeighbours;
    int accuracyPercent;
    int trainTestStrategy;
    int predictionHorizon;
    String trainTestText;
    String featuresJSON;
    String slotsIDJSON;
    String classifiersJSON;
    String attributesJSON;
    HashMap<String, ArrayList<String>> featureData;
    List<Integer> slotsIDData;
    List<Integer> classifiersData;
    List<Integer> attributesData;
    String rawTable; //table with raw parking data
    String preprocessedTable; //table with all the preprocessed data
    String tableName;
    double trainProp;
    String saveIn;
    String modelName;
    String settingsType; // Type either preprocess or training


    Settings(String propertiesFile, Properties props) throws IOException, ParseException {
        InputStream input = ModelTrainer.class.getClassLoader().getResourceAsStream(propertiesFile);
        properties = props;

        // parameters
        dbUri = getSettingAsString("dbUri", false);
        dbUsername = getSettingAsString("dbUsername", false);
        dbPassword = getSettingAsString("dbPassword", false);
        developer = getSettingAsString("developer", false);
        slotsIDJSON = getSettingAsString("slotsIDs", false);
        classifiersJSON = getSettingAsString("classifiers", true);
        attributesJSON = getSettingAsString("attributes", true);
        settingsType = getSettingAsString("type", false);
        slotsIDData = parseStringToIntList(slotsIDJSON);
        if  (slotsIDJSON != null && classifiersJSON != null && attributesJSON != null){
            classifiersData = parseStringToIntList(classifiersJSON);
            attributesData = parseStringToIntList(attributesJSON);
        }
        parkingId = getSettingAsInt("parkingId", false);
        periodMinutes = getSettingAsInt("periodMinutes", false);
        trainingWeeks = getSettingAsInt("trainingWeeks", true);
        saveIn = getSettingAsString("saveIn", false);
        modelName = getSettingAsString("modelName", true);
        trainTestStrategy = getSettingAsInt("trainTestStrategy", true);
        if (trainTestStrategy == 0) {
            trainTestText = "Test and Train Data Mixed";
        } else {
            trainTestText = "Test Data after Train Data";
        }
        preprocessedTable = getSettingAsString("preprocessedTable", true); //Table with preprocessed data
        tableName = getSettingAsString("tableName", false); // Table to save to
        predictionHorizon = getSettingAsInt("predictionHorizon", true);
        rawTable = getSettingAsString("rawTable", true); // Table with raw parking data
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
     * Parse String input into a Integer List
     *
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
