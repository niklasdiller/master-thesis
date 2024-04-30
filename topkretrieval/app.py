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
    perMinList = data["perMin"]
    perMinVars = ', '.join(['%s'] * len(perMinList)) # Join single strings for each element in perMin list for the SQL statement
    predHorList = (data["predHor"])
    predHorVars = ', '.join(['%s'] * len(predHorList)) # Join single strings for each element in predHorList for the SQL statement
    perfMetric = (data["perfMetric"]) # Metric that should be used to calculate the performance score
    k = int(data["k"]) #Number of objects that should be returned to client
    weight = float(data["accWeight"]) #Importance of Performance in comparison to Resource Awareness; Value [0-1]
    algorithm = data["algorithm"] #The algorithm that should be called. Possible values: fagin, threshold, naive.
   
   # Execute SQL Statement
    with connection:
        with connection.cursor() as cursor:
            if predHorList == [] or predHorList == 0: #SQL statement without filter for predHor
                cursor.execute(FILTER_MODELS_NO_PREDHOR.format(perMinVars), (pID,) + tuple(perMinList))
            else:cursor.execute(FILTER_MODELS.format(perMinVars, predHorVars), (pID,) + tuple(perMinList + predHorList))

            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'period_minutes', 'accuracydt', 'accuracyrf', 'accuracylr', 'accuracyknn', '2'])
    
    if df.size == 0: #If no model match the requirements
        raise Exception ("No models found with specified metrics.")   

    df = reshape_perf_table(df, perfMetric) #Call function that summarizes the accuracy for acc
    df = count_attributes(df) #Call function that counts the number of attributes of each model

    if perfMetric == "acc":
        df = normalize(df, '1', rev = False)
    else : # If error metric is chosen, reverse the normalization (Low error value is better)
        df = normalize(df, '1', rev = True)
    df = normalize(df, '2', rev = True)

    df = df.sort_values(by='1', ascending=False, na_position='first') #Sort with highest performance first
    #print(df.head)

    #Slicing Table:
    df_perf = df.drop(columns=['2']) #Performance Table
    df_reaw = df.drop(columns=['1']) #Resource Awareness Table

    df_perf = df_perf.sort_values(by='1', ascending=False, na_position='first') #Sort with highest performance first
    df_reaw = df_reaw.sort_values(by='2', ascending=False, na_position='first') #Sort with least numner of attributes first
    df_dict = {}
    df_dict.update([('1', df_perf), ('2', df_reaw)])

    # Call the desired algrotihm:
    match algorithm:
        case 'fagin':
            result = convert_to_json(fagin_topk(df_dict, weight, k), False, perfMetric)
        case 'threshold':
            result = convert_to_json(threshold_topk(df_dict, weight, k), False, perfMetric)
        case 'naive':
            result = convert_to_json(naive_topk(df, weight, k), False, perfMetric)
        case _:
            raise Exception ("Not a valid algorithm! Try 'naive', 'fagin', or 'threshold'.")

    return result, 200


#Top K Model Sets
@app.post("/api/topk/modelsets")
def topkmodelsets():

    #TODO: unify vairblae names: camelcase or _? change json-names. etc

    data = request.get_json()
    pID = int(data["pID"])
    if pID != 38 and pID != 634:
        raise Exception ("Not a valid parking lot ID. Try 38 or 634.")
    perMinList = data["perMin"]
    perMinVars = ', '.join(['%s'] * len(perMinList)) # Join single strings for each element in perMin list for the SQL statement
    predHorList = (data["predHor"])
    if len(predHorList) == 1:
        raise Exception ("For modelset retrieval at least 2 prediction horizons must be selected!")
    predHorVars = ', '.join(['%s'] * len(predHorList)) # Join single strings for each element in predHorList for the SQL statement
    perfMetric = (data["perfMetric"]) # Metric that should be used to calculate the performance score
    k = int(data["k"]) #Number of models that should be used per prediction Horizon to create modelsets
    if data["n"] == "max": #Number of modelsets that should be returned to the client
        n = "max"
    else: n = int(data["n"])
    weight1 = float(data["accWeight"]) #Importance of Performance in comparison to Resource Awareness; Value [0-1]
    weight2 = float(data["MSSWeight"]) #Importance of Modelset Score in comparison to Query Sharing Level; Value [0-1]
    algorithm = data["algorithm"] #The algorithm that should be called. Possible values: fagin, threshold, naive.
    combineSameFeatures = (data["combineSameFeatures"]) #Indicates, whether modelsets should only be generated if the models have the same features
    calculateQSL = data["calculateQSL"]
    with connection:
        with connection.cursor() as cursor:
            if predHorList == [] or predHorList == 0:
                raise Exception ("One or multiple Prediction Horizons must be provided!")
            else:
                #Putting the variables into the statement
                cursor.execute(FILTER_MODELS_MODELSETS.format(perMinVars, predHorVars), (pID,) + tuple(perMinList + predHorList))
            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'prediction_horizon', 'period_minutes', 'accuracydt', 'accuracyrf', 'accuracylr', 'accuracyknn', '2'])
    
    if df.size == 0: #If no model match the requirements
        raise Exception ("No models found with specified metrics.")   
    df = reshape_perf_table(df, perfMetric) #Puts the performance metric into an atomic
    df = count_attributes(df) #Counts the number of attributes of each model

    df = normalize(df, '1', rev = False)
    df = normalize(df, '2', rev = True)

    df = df.sort_values(by='1', ascending=False, na_position='first') #Sort with highest performance first
    print(df.head)
    df_dict = {}
    df_dict_naive = {} #Dict of DF for the naive algorithm: No splitting done for performance/attributes

    #Splitting Table into smaller tables for each predHor value * 2 (performance and attributes)
    # df_dict with df_metric as value per predHor, which holds 2 more df 
    for key in predHorList: 
        df_predHor = df.drop(df[df.prediction_horizon != int(key)].index)
        key = "predHor"+str(key)

        df_metric = {}
        df_perf = df_predHor.drop(columns=['2'])
        df_reaw = df_predHor.drop(columns=['1'])
        #df_reaw = df_predHor.drop(columns=['performance'])

        df_perf = df_perf.sort_values(by='1', ascending=False, na_position='first') #Sort with highest performance first
        df_reaw = df_reaw.sort_values(by='2', ascending=False, na_position='first') #Sort with least numner of attributes first

        df_metric.update([('1', df_perf), ('2', df_reaw)])
        df_dict.update({key: df_metric}) # Adding into dict each df containing a unique predHor
        df_dict_naive.update({key:df_predHor}) # Fill dict for naive algortithm

    print("First TopK:")
    # Call the desired algrotihm:
    result = []
    for key in df_dict:
        match algorithm:
            case 'fagin':
                result.append(fagin_topk(df_dict.get(key), weight1, k))
            case 'threshold':
                #result = threshold_topk(df_dict, weight, k)
                result.append(threshold_topk(df_dict.get(key), weight1, k))
            case 'naive':
                result.append(naive_topk(df_dict_naive.get(key), weight1, k))
            case _:
                raise Exception ("Not a valid algorithm! Try 'naive', 'fagin', or 'threshold'.")

    combinations_raw = create_combinations(result, combineSameFeatures) # Create modelsets by combining models
    combinations = [] # List that contains JSON convertable datatypes only

    # Get the best modelset by calculating overall score
    for cur_combi in combinations_raw:
            modelset = OrderedDict([('Modelset Number', None), ('Models', {}), ( 'Modelset Score', None), ('Query Sharing Level', None)]) #Define dict to turn into JSON
            modelset_score = 0
            for index, model in enumerate(cur_combi):
                model_dict = model.to_dict() # Converting the series into dict
                modelname = 'Model'+str(index+1)
                modelset['Models'][modelname] = model_dict
                modelset_score += model['score'] #Calculate overall score for each combination

            modelset_score = round(modelset_score, 2)
            modelset.update({'Modelset Score': modelset_score}) #Append the overall score to each combination
            combinations.append(modelset)
            #print("MSS ", modelset_score)
    
    combinations= sorted(combinations, key = lambda d: d['Modelset Score'], reverse=True) # Sort by value of Modelset Score

    if n != "max": #If user put "max", all the created modelsets will be displayed.
        del combinations[n:] #Delete every object from n to end of list
    else:
        n = len(combinations) # If max was set, return the total number of combinations to the client

    for index,modelset in enumerate(combinations): 
        modelset.update({'Modelset Number' : index+1 }) #Give each modelset a number
        qsl_list = []

        # Calculate Query Sharing Level
        cur_models = itertools.combinations(modelset['Models'].items(), 2) # Generate each combination of 2 models
        for model1, model2 in cur_models: 
            
            #model[1] is the model specs; split('-')[1] is the second part split by '-', so the features
            sameFeauters =  model1[1]["model_name"].split('-')[1] == model2[1]["model_name"].split('-')[1] #Check if features are the same
            samePerMin =  model1[1]["period_minutes"] == model2[1]["period_minutes"] #Check if periodMinutes are the same
            
            if sameFeauters and samePerMin: # Level 2: Same features
                qsl_list.append(2) 

            elif samePerMin: # Level 1: Same segmentation (=perMin)
                qsl_list.append(1) 

            else:  # Level 0: No sharing possible
                qsl_list.append(0)

        match calculateQSL:
            case 'max': # Max level will be chosen as overall QSL
                modelset.update({'Query Sharing Level' : max(qsl_list)})
            case 'min': # Min level wil be chosen for overall QSL
                modelset.update({'Query Sharing Level' : min(qsl_list)})
            case 'avg': # Average of all levels will be calculated
                level_sum = sum(qsl_list)
                level_avg = level_sum / len(qsl_list)
                level_avg = round(level_avg, 2)
                modelset.update({'Query Sharing Level' : level_avg})
            case _:
                raise Exception ("Not a valid QSL calculation! Try 'max', 'min', or 'avg'.")


    #print(combinations)


    #Slicing Tables
    df_fromOD = pd.DataFrame(combinations) # Create DF from Ordered Dictionary
    df_fromOD = df_fromOD.rename(columns={"Modelset Score": "1", "Query Sharing Level": "2"})

    df_QSL = df_fromOD.drop(columns=['1'])
    df_MSS = df_fromOD.drop(columns=['2'])

    #print("Head", df_QSL.head)
    df_QSL = df_QSL.sort_values(by='2', ascending=False, na_position='first')
    df_MSS = df_MSS.sort_values(by='1', ascending=False, na_position='first')
    df_dict = {}
    df_dict.update([('QSL', df_QSL), ('MSS', df_MSS)])

    print("Second TopK:")
     # Call the desired algrotihm:
    match algorithm:
        case 'fagin':
            result = fagin_topk(df_dict, weight2, n)
        case 'threshold':
            result = threshold_topk(df_dict, weight2, n)
        case 'naive':
            result = naive_topk(df_fromOD, weight2, n)

    #print("Result", result)

    result_json= convert_to_json(result, True, perfMetric)
    return (result_json), 200