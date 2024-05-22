from tokenize import String
import pandas as pd
import re
import json
import itertools

def normalize (df: pd.DataFrame, col: str, rev: bool):
    normCol = [col]
    if not rev: #E.g. for performance data: High values -> high score 
        df[normCol] = df[normCol].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    else: #E.g. to have models with low number of attributes/low error metric have a high RA/Perf score: High value -> low score
        df[normCol] = df[normCol].apply(lambda x: ((x - x.min()) / (x.max() - x.min()) -1) * -1) 
        #Replace -0.0 with 0.0 for reverse normalization
        for ind in df.index:
            if df.at[ind, col] == (-0.0):
                df.at[ind, col] += 0
    return df

    
def create_combinations (resultList: list, combineSameFeatures: bool):
    if combineSameFeatures == True: # Only combine models to modelsets that have the same features
        combinations = []
        for models_per_pH in itertools.product(*resultList): #For all models per Prediction Horizon
            cur_combi = []
            #print(models_per_pH)
            features_set = set() # A set to keep track of the features in one combination
            for model in models_per_pH: #For all models 
                model_name = model.get("model_name")
                features = model_name.split('-')[1] # get the featues 
                features_set.add(features)
                cur_combi.append(model)
            if len(features_set) == 1: # As a set does not contain duplicate values length 1 means only the same features in the combination
                combinations.append(cur_combi) # Only if same features, the combination is a valid modelset
        return combinations

    else: #Create all possible combinations of models for modelsets
        combinations = []
        for models_per_pH in itertools.product(*resultList): #For all models per Prediction Horizon
            cur_combi = []
            for model in models_per_pH: #For all models 
                cur_combi.append(model)
            combinations.append(cur_combi)
        return combinations

 
def convert_to_json (result, is_modelset:bool, perf_metric): #Convert result list into JSON format
    json_list = []
    perf_metric_string = str(perf_metric) # String formatting 
    if perf_metric_string == "acc":
        perf_metric_string = "Accuracy"
    else: 
        perf_metric_string = perf_metric_string.upper() 
    # If single models are retrieved
    if is_modelset == False:
        count = 1
        for row in result:
            modelnumber = "model"+str(count) #For model identifier in reply
            model_id = int(row["model_id"]) #Convert serial to int
            dict = {
                "Model Number": modelnumber,
                "Model Specs": [ 
                    {
                        "Model ID": model_id,
                        "Model Name": row["model_name"],
                        "Window Size": int(row["window_size"]),
                        perf_metric_string: row["perfMetric"],
                        "Performance Score": row["1"],
                        "Resource Awareness Score": row["2"],
                        "Model Score": row["score"]
                    }
                ]   
            }

            json_list.append(dict)
            count += 1
        json_result = json.dumps(json_list, indent=4)   
        return json_result


    # If modelsets are retrieved
    elif is_modelset == True: #If modelsets are queried
        for i, modelset in enumerate(result):
            modelsetnumber = "Modelset"+str(i+1)
            modelset_dict = {
                "Modelset Number" : modelsetnumber,
                "Modelset Score" : float(modelset.get('score')), #Making sure each value is of type float
                "Aggregated Model Score" : float(modelset.get('1')), 
                "Query Sharing Level" : float(modelset.get('2'))
                
            }

            for modelname, model in modelset['Models'].items(): 
                model_dict = {
                    "Model Specs": [
                        {
                            "Model ID": model.get('model_id'),
                            "Model Name": model.get("model_name"),
                            "Prediction Horizon": model.get('prediction_horizon'),
                            "Window Size": model.get('window_size'),
                            perf_metric_string: model.get('perfMetric'),
                            "Performance Score": round(model.get("1"), 2),
                            "Resource Awareness Score": model.get("2"),
                            "Model Score": model.get("score")
                        }
                    ]
                }

                modelset_dict.update({modelname:model_dict})

            json_list.append(modelset_dict)
        json_result = json.dumps(json_list, indent=4)   
        return json_result


def round_result(obj): #Round attribute, performance and overall score of each model
    obj["score"] = round(obj["score"], 2)
    obj["1"] = round(obj["1"], 2)
    obj["2"] = round(obj["2"], 2)
    return obj

def round_result_df(df):
    df['1'] = df['1'].astype(float).round(2)
    df['2'] = df['2'].astype(float).round(2)
    df['score'] = df['score'].astype(float).round(2)
    return df

def count_attributes(df: pd.DataFrame): #Count the number of attributes used in each model
    for ind in df.index:
        val = df.at[ind, '2']
        numAttr = len(val.split(', '))
        df.at[ind, '2'] = numAttr # replace the actual attirbute values with the number of attributes used
        df = penalize_small_window_size(df, ind) 
    return df

def penalize_small_window_size(df, ind):
    if df.at[ind, 'window_size'] == 1:
        df.at[ind, '2'] += 3 #If model is using winSize = 1, penalize the RA score by adding 3. Customizable value
    return df

def get_perf_metric (df: pd.DataFrame, perf_metric):
    df.insert(2, 'perfMetric', None) # New column that shows the absolute value of the performance metric
    numberOfCols = len(df.columns) 
    df.insert(numberOfCols-1, '1', None) # New column that swill show the score later

    match perf_metric:
        case "acc":
            for ind in df.index: #Get the desired performance metric
                val = df.at[ind, 'accuracy'] #Specify which metric should be considered here
                df.at[ind, 'perfMetric'] = val 
                df.at[ind, '1'] = val 
        case "mae":
            for ind in df.index:
                val = df.at[ind, 'mae']
                df.at[ind, 'perfMetric'] = val 
                df.at[ind, '1'] = val 
        case "mse":
            for ind in df.index:
                val = df.at[ind, 'mse']
                df.at[ind, 'perfMetric'] = val
                df.at[ind, '1'] = val
        case "rmse":
            for ind in df.index:
                val = df.at[ind, 'rmse']
                df.at[ind, 'perfMetric'] = val 
                df.at[ind, '1'] = val
        case _:
            raise Exception ("Not a valid performance metric! Try 'acc', 'mae', 'mse' or 'rmse'.")
    df = df.drop(columns=['accuracy', 'mae', 'mse', 'rmse'])
    return df


# Obsolete functions, now that the performance metrics are saved atomically
'''
def reshape_perf_table(df: pd.DataFrame, perfMetric):
    df = df.rename(columns = {'accuracydt':'1'})
    df.insert(2, 'perfMetric', None) # New column that shows the absolute value of the performance metric
    for ind in df.index: #Collect the performance values into a single one
        if df.at[ind, '1'] == 'no classifier':
            if df.at[ind, 'accuracyrf'] != 'no classifier':
                df.at[ind, '1'] = df.at[ind, 'accuracyrf']
            elif df.at[ind, 'accuracylr'] != 'no classifier':
                 df.at[ind, '1'] = df.at[ind, 'accuracylr']
            else: 
                 df.at[ind, '1'] = df.at[ind, 'accuracyknn']
        completeString = df.at[ind, '1']

        match perfMetric:
            case "acc":
                val = get_acc(completeString) #Specify which metric should be considered here
            case "mae":
                val = get_mae(completeString)
            case "mse":
                 val = get_mse(completeString)
            case "rmse":
                val = get_rmse(completeString)
            case _:
                raise Exception ("Not a valid performance metric! Try 'acc', 'mae', 'mse' or 'rmse'.")
        df.at[ind, 'perfMetric'] = val #Set float value for the score later
        df.at[ind, '1'] = val #Set float value for the absolute performance metric for later display
    df = df.drop(columns=['accuracyrf', 'accuracylr', 'accuracyknn']) #Remove redundant columns
    return df

def get_acc(str): #Accuracy
    pattern = r'Correctly predicted: (\d+\.\d+)%'
    match = re.search(pattern, str)
    if match:
        val = float(match.group(1)) #Get relevant substring and convert into float
        return val
    else:
        return None

def get_mae(str): #Mean Absolute Error
    pattern = r'MAE: (\d+\.\d+)'
    match = re.search(pattern, str)
    if match:
        val = float(match.group(1))
        return val
    else:
        return None

def get_mse(str): #Mean Squared Error
    pattern = r'MSE: (\d+\.\d+)'
    match = re.search(pattern, str)
    if match:
        val = float(match.group(1))
        return val
    else:
        return None

def get_rmse(str): #Root Mean Squared Error
    pattern = r'RMSE: (\d+\.\d+)'
    match = re.search(pattern, str)
    if match:
        val = float(match.group(1))
        return val
    else:
        return None
'''