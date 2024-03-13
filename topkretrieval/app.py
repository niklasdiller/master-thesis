from tokenize import String
from flask import Flask, request, jsonify
import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
import json

load_dotenv()

app = Flask(__name__)
url = os.getenv("DATABASE_URL")
connection = psycopg2.connect(url)

SELECT_MODEL =  """SELECT model_id, model_name, developer, created_time, attributes, classifiers, model_size_in_bytes, 
                randomforestmaxdepth, kneighbours , accuracydt, accuracyrf, accuracylr, accuracyknn
                FROM niklas_trained_models 
                WHERE model_name = %s
                ORDER BY model_id desc
                LIMIT %s;"""

SELECT_MODEL_DIRECT =  """SELECT model_id, model_name, developer, created_time, attributes, classifiers, model_size_in_bytes, 
                randomforestmaxdepth, kneighbours , accuracydt, accuracyrf, accuracylr, accuracyknn
                from niklas_trained_models 
                where model_name = %s
                order by model_id desc;"""

FILTER_MODELS =  """SELECT model_id, model_name, accuracydt, accuracyrf, accuracylr, accuracyknn, attributes
                FROM niklas_trained_models
                WHERE parking_id = %s and period_minutes = %s"""

# TABLE_ATT = """SELECT model_id, model_name, attributes
#                 FROM niklas_trained_models
#                 WHERE parking_id = %s and period_minutes = %s"""


@app.get("/") #Define endpoint for home screen
def home():
    return "Hello, world!"

# Select a model with a given name
@app.post("/api/select")
def select():
    data = request.get_json()
    model_name = data["model_name"]
    limit = data["limit"]
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_MODEL, (model_name, limit ))
            model = cursor.fetchall()
    return {"model": model, "message": "Data selected!"}, 200

# Selet a model with a given name directly using the URL
@app.get("/api/model/<model_name>")
def select_direct(model_name):
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_MODEL_DIRECT, (model_name, ))
            model = cursor.fetchall()
    if model == []:
        return {"message": "No such model found in database."}, 404
    return {"model": model, "message": "Data directly selected!"}, 200

#Top K
@app.post("/api/topk")
def topk():
    data = request.get_json()
    pID = data["pID"]
    perMin = data["perMin"]
    k = int(data["k"]) #Number of objects that should be returned to client
    weight = data["accWeight"] #Importance of Accuracy in comparison to Resource Awareness; Value [0-1]
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(FILTER_MODELS, (pID, perMin))
            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'accuracydt', 'accuracyrf', 'accuracylr', 'accuracyknn', 'attributes'])
            
    df = reshape_perf_table(df) #Call function that summarizes the accuracy for acc
    df = count_attributes(df) #Call function that counts the number of attributes of each model

    #Normalize Performance data
    normColumnPerf = ['performance']
    df[normColumnPerf] = df[normColumnPerf].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    #Normalize RA data and have models with low number of attributes have high RA score
    normColumnRA = ['attributes']
    df[normColumnRA] = df[normColumnRA].apply(lambda x: ((x - x.min()) / (x.max() - x.min()) -1) * -1) 

    #Replace -0.0 with 0.0
    for ind in df.index:
        if df.at[ind, 'attributes'] == (-0.0):
            df.at[ind, 'attributes'] += 0

    df = df.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
    print(df.head)

    #Performance Table:
    df_perf = df.drop(columns=['attributes'])
    #Resource Awareness Table:
    df_reaw = df.drop(columns=['performance'])

    df_perf = df_perf.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
    df_reaw = df_reaw.sort_values(by='attributes', ascending=False, na_position='first') #Sort with least numner of attributes first

    #TODO: Determine what Algorithm should be called
    result = naive_topk(df, weight, k)

    #TODO: Round performance, attributes and score

    return result

def convert_to_json (result: list): #Convert result list into JSON format
    json_list = []
    count = 1
    for row in result:
        modelnumber = "model"+str(count) #For model identifier in reply
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

def naive_topk (df: pd.DataFrame, weight: float, k: int):
    result = []
    weight = float(weight)
    for ind in df.index:
       #Compute Score and put in new column
       score = (df.at[ind, 'performance'] * weight) + (df.at[ind, 'attributes'] * (1-weight))
       df.at[ind, 'score'] = score

    df = df.sort_values(by='score', ascending=False, na_position='first')  #Sort for score

    for ind in range(k):
        result.append(df.iloc[ind])  

    for ind in range(len(result)):
        round_result(result[ind])

    return convert_to_json(result)

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