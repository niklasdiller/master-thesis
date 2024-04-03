from tokenize import String
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd
import re
import json

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

def convert_to_json (result): #Convert result list into JSON format
    json_list = []
    count = 1
    for row in result:
        modelnumber = "model"+str(count) #For model identifier in reply
        #print(result)
        if type(result) == pd.DataFrame: #is result is a dataframe
            model_id = int(result.at[count-1, 'model_id'])
            dict = {
                "modelnumber": modelnumber,
                "model_specs": [ 
                    {
                        "model_id": model_id,
                        "model_name": result.at[count-1, "model_name"],
                        "performance": result.at[count-1, "performance"],
                        "attributes": result.at[count-1, "attributes"],
                        "score": result.at[count-1, "score"]
                    }
                ]
            }

        else: # if result is a list
            model_id = int(row["model_id"]) #Convert serial to int
            dict = {
                "modelnumber": modelnumber,
                "model_specs": [ 
                    {
                        "model_id": model_id,
                        "model_name": row["model_name"],
                        "performance": row["performance"],
                        "attributes": row["attributes"],
                        "score": row["score"]
                    }
                ]   
            }
        json_list.append(dict)
        count += 1
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
        df.at[ind, 'attributes'] = numAttr
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