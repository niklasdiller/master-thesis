from tokenize import String
import pandas as pd
import re
import json
from collections import OrderedDict
import itertools
import numpy as np

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
                        "Period Minutes": int(row["period_minutes"]),
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
    elif isModelset == True: #If modelsets are queried
        for i, modelset in enumerate(result):
            modelsetnumber = "Modelset"+str(i+1)
            modelset_dict = {
                "Modelsetnumber" : modelsetnumber,
                "Overall Score" : float(modelset.get('score')),
                "Modelset Score" : float(modelset.get('1')), #Making sure each value is of type float
                "Query Sharing Level" : float(modelset.get('2'))
                
            }

            for modelname, model in modelset['Models'].items(): 
                model_dict = {
                    "Model Specs": [
                        {
                            "Model ID": model.get('model_id'),
                            "Model Name": model.get("model_name"),
                            "Prediction Horizon": model.get('prediction_horizon'),
                            "Period Minutes": model.get('period_minutes'),
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
    obj[-1] = round(obj[-1], 2)
    obj[-2] = round(obj[-2], 2)
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
    if df.at[ind, 'period_minutes'] == 1:
        df.at[ind, '2'] += 3 #If model is using perMin = 1, penalize the RA score by adding 3. Customizable value
    return df


def reshape_perf_table(df: pd.DataFrame):
    df = df.rename(columns = {'accuracydt':'1'})
    for ind in df.index: #Collect the performance values into a single one
        if df.at[ind, '1'] == 'no classifier':
            if df.at[ind, 'accuracyrf'] != 'no classifier':
                df.at[ind, '1'] = df.at[ind, 'accuracyrf']
            elif df.at[ind, 'accuracylr'] != 'no classifier':
                 df.at[ind, '1'] = df.at[ind, 'accuracylr']
            else: 
                 df.at[ind, '1'] = df.at[ind, 'accuracyknn']
        str = df.at[ind, '1']
        val = get_acc(str) #Specify which metric should be considered here
        df.at[ind, '1'] = val #Set float value
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