from flask import Flask, request
import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import re

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
def selectDirect(model_name):
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_MODEL_DIRECT, (model_name, ))
            model = cursor.fetchall()
    if model == []:
        return {"message": "No such model found in database."}, 404
    return {"model": model, "message": "Data directly selected!"}, 200

#Top K Naive
@app.post("/api/topk")
def topk():
    data = request.get_json()
    pID = data["pID"]
    perMin = data["perMin"]
    k = data["k"]
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(FILTER_MODELS, (pID, perMin))
            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'accuracydt', 'accuracyrf', 'accuracylr', 'accuracyknn', 'attributes'])
            #Performance Table:
            df_perf = df.drop(columns=['attributes'])
            #Resource Awareness Table:
            df_reaw = df.drop(columns=['accuracydt', 'accuracyrf', 'accuracylr', 'accuracyknn'])

    df_perf = reshapePerfTable(df_perf) #Call function that summarizes the accuracy for acc
    df_reaw = countAttributes(df_reaw) #Call function that counts the number of attributes of each model

    df_perf = df_perf.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
    df_reaw = df_reaw.sort_values('attributes', na_position='first') #Sort with least numner of attributes first
    
    print(df_perf.head)
    print(df_reaw.head)
    return {"response": "OK"}

def countAttributes(df): #Count the number of attributes used in each model
    for ind in df.index:
        val = df.at[ind, 'attributes']
        numAttr = len(val.split(', '))
        df.at[ind, 'attributes'] = numAttr
    return df

def reshapePerfTable(df):
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
        val = getAcc(str) #Specify which metric should be considered here
        df.at[ind, 'performance'] = val #Set float value
    df = df.drop(columns=['accuracyrf', 'accuracylr', 'accuracyknn']) #Remove redundant columns
    return df

def getAcc(str): #Accuracy
    pattern = r'Correctly predicted: (\d+\.\d+)%'
    match = re.search(pattern, str)
    val = float(match.group(1)) #Get relevant substring and convert into float
    return val

def getMAE(str): #Mean Absolute Error
    pattern = r'MAE: (\d+\.\d+)'
    match = re.search(pattern, str)
    val = float(match.group(1))
    return val

def getMSE(str): #Mean Squared Error
    pattern = r'MSE: (\d+\.\d+)'
    match = re.search(pattern, str)
    val = float(match.group(1))
    return val

def getRMSE(str): #Root Mean Squared Error
    pattern = r'RMSE: (\d+\.\d+)'
    match = re.search(pattern, str)
    val = float(match.group(1))
    return val
