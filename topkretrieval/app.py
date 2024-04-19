from email import utils
from tokenize import String
from flask import Flask, request, jsonify
import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from collections import OrderedDict

pd.options.mode.chained_assignment = None  # default='warn' # Ignores warnings regarding chaning values to df copy

from utils.helper import *
from utils.sql import *
from utils.algorithms import *

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

    data = request.get_json()
    pID = int(data["pID"])
    if pID != 38 and pID != 634:
        raise Exception ("Not a valid parking lot ID. Try 38 or 634.")
    perMin = data["perMin"]
    predHor = int(data["predHor"])
    k = int(data["k"]) #Number of objects that should be returned to client
    weight = float(data["accWeight"]) #Importance of Performance in comparison to Resource Awareness; Value [0-1]
    algorithm = data["algorithm"] #The algorithm that should be called. Possible values: fagin, threshold, naive.
    with connection:
        with connection.cursor() as cursor:
            if predHor == 0:
                cursor.execute(FILTER_MODELS_NO_PREDHOR, (pID, perMin)) #SQL statement without filter for predHor
            else: cursor.execute(FILTER_MODELS, (pID, perMin, predHor))
            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'accuracydt', 'accuracyrf', 'accuracylr', 'accuracyknn', 'attributes'])
    
    if df.size == 0: #If no model match the requirements
        raise Exception ("No models found with specified metrics.")   

    df = reshape_perf_table(df) #Call function that summarizes the accuracy for acc
    df = count_attributes(df) #Call function that counts the number of attributes of each model

    df = normalize(df, 'performance', rev = False)
    df = normalize(df, 'attributes', rev = True)

    df = df.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
    #print(df.head)

    #Slicing Table:
    df_perf = df.drop(columns=['attributes']) #Performance Table
    df_reaw = df.drop(columns=['performance']) #Resource Awareness Table

    df_perf = df_perf.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
    df_reaw = df_reaw.sort_values(by='attributes', ascending=False, na_position='first') #Sort with least numner of attributes first
    df_dict = {}
    df_dict.update([('performance', df_perf), ('attributes', df_reaw)])

    # Call the desired algrotihm:
    match algorithm:
        case 'fagin':
            result = convert_to_json(fagin_topk(df_dict, weight, k), False)
        case 'threshold':
            result = convert_to_json(threshold_topk(df_dict, weight, k), False)
        case 'naive':
            result = convert_to_json(naive_topk(df, weight, k), False)
        case _:
            raise Exception ("Not a valid algorithm! Try 'naive', 'fagin', or 'threshold'.")

    return result


#Top K Model Sets
@app.post("/api/topk/modelsets")
def topkmodelsets():

    data = request.get_json()
    pID = int(data["pID"])
    if pID != 38 and pID != 634:
        raise Exception ("Not a valid parking lot ID. Try 38 or 634.")
    perMin = data["perMin"]
    predHorList = (data["predHor"])
    predHorVars = ', '.join(['%s'] * len(predHorList)) # Join single strings for each element in predHorList for the SQL statement
    k = int(data["k"]) #Number of models that should be used per prediction Horizon to create modelsets
    if data["n"] == "max": #Number of modelsets that should be returned to the client
        n = "max"
    else: n = int(data["n"])
    weight = float(data["accWeight"]) #Importance of Performance in comparison to Resource Awareness; Value [0-1]
    algorithm = data["algorithm"] #The algorithm that should be called. Possible values: fagin, threshold, naive.
    combineSameFeatures = (data["combineSameFeatures"]) #Indicates, whether modelsets should only be generated if the models have the same features
    with connection:
        with connection.cursor() as cursor:
            if predHorList == [] or predHorList == 0:
                raise Exception ("One or multiple Prediction Horizons must be provided!")
            else:
                #cursor.execute(FILTER_MODELS_MODELSETS, (pID, perMin, predHorSQL))
                cursor.execute(FILTER_MODELS_MODELSETS.format(predHorVars), (pID, perMin) + tuple(predHorList))
            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'prediction_horizon', 'accuracydt', 'accuracyrf', 'accuracylr', 'accuracyknn', 'attributes'])
    
    if df.size == 0: #If no model match the requirements
        raise Exception ("No models found with specified metrics.")   

    df = reshape_perf_table(df) #Puts the performance metric into an atomic
    df = count_attributes(df) #Counts the number of attributes of each model

    df = normalize(df, 'performance', rev = False)
    df = normalize(df, 'attributes', rev = True)

    df = df.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
    #print(df.head)
    df_dict = {}
    df_dict_naive = {} #Dict of DF for the naive algorithm: No splitting done for performance/attributes

    #Splitting Table into smaller tables for each predHor value * 2 (performance and attributes)
    # df_dict with df_metric as value per predHor, which holds 2 more df 
    for key in predHorList: 
        df_predHor = df.drop(df[df.prediction_horizon != int(key)].index)
        key = "predHor"+str(key)

        df_metric = {}
        df_perf = df_predHor.drop(columns=['attributes', 'prediction_horizon'])
        df_reaw = df_predHor.drop(columns=['performance', 'prediction_horizon'])
        #df_reaw = df_predHor.drop(columns=['performance'])

        df_perf = df_perf.sort_values(by='performance', ascending=False, na_position='first') #Sort with highest performance first
        df_reaw = df_reaw.sort_values(by='attributes', ascending=False, na_position='first') #Sort with least numner of attributes first

        df_metric.update([('performance', df_perf), ('attributes', df_reaw)])
        df_dict.update({key: df_metric}) # Adding into dict each df containing a unique predHor
        df_dict_naive.update({key:df_predHor}) # Fill dict for naive algortithm


    # Call the desired algrotihm:
    result = []
    for key in df_dict:
        match algorithm:
            case 'fagin':
                result.append(fagin_topk(df_dict.get(key), weight, k))
            case 'threshold':
                #result = threshold_topk(df_dict, weight, k)
                result.append(threshold_topk(df_dict.get(key), weight, k))
            case 'naive':
                result.append(naive_topk(df_dict_naive.get(key), weight, k))
            case _:
                raise Exception ("Not a valid algorithm! Try 'naive', 'fagin', or 'threshold'.")


    combinations = create_combinations(result, combineSameFeatures) # Create modelsets by combining models
    combinations_json = [] # List that contains JSON convertable datatypes only

    # Get the best modelset by calculating overall score
    for cur_combi in combinations:
            modelset = OrderedDict([('Modelset Number', None), ( 'Modelset Score', None), ('Models', {}) ]) #Define dict to turn into JSON
            modelset_score = 0
            for index, model in enumerate(cur_combi):
                model_dict = model.to_dict() # Converting the series into dict
                model_dict.update({"prediction_horizon": predHorList[index]}) # Add prediction horizon as model metric
                modelname = 'Model'+str(index+1)
                modelset['Models'][modelname] = model_dict
                modelset_score += model['score'] #Calculate overall score for each combination

            modelset_score = round(modelset_score, 2)
            modelset.update({'Modelset Score': modelset_score}) #Append the overall score to each combination
            combinations_json.append(modelset)
            #print("MSS ", modelset_score)
    
    combinations_json= sorted(combinations_json, key = lambda d: d['Modelset Score'], reverse=True) # Sort by value of Modelset Score

    if n != "max": #If user put "max", all the created modelsets will be displayed.
        del combinations_json[n:] #Delete every object from n to end of list

    for index,modelset in enumerate(combinations_json): #Give each modelset a number
        modelset.update({'Modelset Number' : index+1 })

    result_json= convert_to_json(combinations_json, True)
    return (result_json)