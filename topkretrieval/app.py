from email import utils
from tokenize import String
from flask import Flask, request, jsonify
import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import numpy as np

import json

from utils.helper import *
from utils.sql import *

load_dotenv()

app = Flask(__name__)
url = os.getenv("DATABASE_URL")
connection = psycopg2.connect(url)

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
    #TODO: Error hanlding: wrong parkingLot ID

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

    df = normalize(df, 'performance', False)
    df = normalize(df, 'attributes', True)

    df = df.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
    print(df.head)

    #Slicing Table here
    #Performance Table:
    df_perf = df.drop(columns=['attributes'])
    #Resource Awareness Table:
    df_reaw = df.drop(columns=['performance'])

    df_perf = df_perf.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
    df_reaw = df_reaw.sort_values(by='attributes', ascending=False, na_position='first') #Sort with least numner of attributes first

    #TODO: Determine what Algorithm should be called
    result = naive_topk(df, weight, k)

    return result

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
