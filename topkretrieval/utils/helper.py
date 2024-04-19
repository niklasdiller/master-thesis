from tokenize import String
import pandas as pd
import re
import json
from collections import OrderedDict
import itertools

def normalize (df: pd.DataFrame, col: str, rev: bool):
    normCol = [col]
    if not rev: #E.g. for performance data: High values -> high score 
        df[normCol] = df[normCol].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    else: #E.g. to have models with low number of attributes a high RA score: High value -> low score
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
        print("Length", len(combinations))
        return combinations


def convert_to_json (result, isModelset:bool): #Convert result list into JSON format
    json_list = []
    # If single models are retrieved
    if isModelset == False:
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
                        "Performance": row["performance"],
                        "Attributes": row["attributes"],
                        "Score": row["score"]
                    }
                ]   
            }

            json_list.append(dict)
            count += 1
        json_result = json.dumps(json_list, indent=4)   
        return json_result


    # If modelsets are retrieved
    elif isModelset == True: #If modelsets are queried
        for i, modelset in enumerate(result):
            modelsetnumber = "Modelset"+str(i+1)
            modelset_dict = {
                "Modelsetnumber" : modelsetnumber,
                "Modelset Score" : modelset.get('Modelset Score')
            }

            for modelname, model in modelset['Models'].items(): 
                model_dict = {
                    "Model Specs": [
                        {
                            "Model ID": model.get('model_id'),
                            "Model Name": model.get("model_name"),
                            "Prediction Horizon": model.get('prediction_horizon'),
                            "Performance": model.get("performance"),
                            "Attributes": model.get("attributes"),
                            "Score": model.get("score")
                        }
                    ]
                }

                modelset_dict.update({modelname:model_dict})

            json_list.append(modelset_dict)
        json_result = json.dumps(json_list, indent=4)   
        return json_result


def round_result(obj): #Round attribute, performance and overall score of each model
    obj["score"] = round(obj["score"], 2)
    obj["performance"] = round(obj["performance"], 2)
    obj["attributes"] = round(obj["attributes"], 2)
    return obj

def round_result_df(df):
    df['performance'] = df['performance'].astype(float).round(2)
    df['attributes'] = df['attributes'].astype(float).round(2)
    df['score'] = df['score'].astype(float).round(2)
    return df

def count_attributes(df: pd.DataFrame): #Count the number of attributes used in each model
    for ind in df.index:
        val = df.at[ind, 'attributes']
        numAttr = len(val.split(', '))
        df.at[ind, 'attributes'] = numAttr # replace the actual attirbute values with the number of attributes used
    return df

def reshape_perf_table(df: pd.DataFrame):
    df = df.rename(columns = {'accuracydt':'performance'})
    for ind in df.index: #Collect the performance values into a single one
        if df.at[ind, 'performance'] == 'no classifier':
            if df.at[ind, 'accuracyrf'] != 'no classifier':
                df.at[ind, 'performance'] = df.at[ind, 'accuracyrf']
            elif df.at[ind, 'accuracylr'] != 'no classifier':
                 df.at[ind, 'performance'] = df.at[ind, 'accuracylr']
            else: 
                 df.at[ind, 'performance'] = df.at[ind, 'accuracyknn']
        str = df.at[ind, 'performance']
        val = get_acc(str) #Specify which metric should be considered here
        df.at[ind, 'performance'] = val #Set float value
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