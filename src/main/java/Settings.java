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
    int windowSize;
    int trainingWeeks;
    int randomForestMaxDepth;
    int kNeighbours;
    int accuracyPercent;
    int trainTestStrategy;
    int predictionHorizon;
    String trainTestText;
    String featuresJSON;
    String spaceIDJSON;
    String classifiersJSON;

    HashMap<String, ArrayList<String>> featureData;
    List<Integer> spaceIDData;
    List<Integer> classifiersData;
    List<Integer> featuresData;
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
        spaceIDJSON = getSettingAsString("spaceIDs", false);
        classifiersJSON = getSettingAsString("classifiers", true);
        featuresJSON = getSettingAsString("features", true);
        settingsType = getSettingAsString("type", false);
        spaceIDData = parseStringToIntList(spaceIDJSON);
        if  (spaceIDJSON != null && classifiersJSON != null && featuresJSON != null){
            classifiersData = parseStringToIntList(classifiersJSON);
            featuresData = parseStringToIntList(featuresJSON);
        }
        parkingId = getSettingAsInt("parkingId", false);
        windowSize = getSettingAsInt("windowSize", false);
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
        List<Integer> spaceIDList = new ArrayList<>();

        if (!cleanedString.isBlank()) {
            spaceIDList = Arrays.stream(cleanedString.split("\\s"))
                    .map(Integer::parseInt)
                    .collect(Collectors.toList());
        }
        return spaceIDList;
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
