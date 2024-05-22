from tokenize import String
from flask import Flask, request, jsonify
import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
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
    #TODO: Set up a simple homepage with instructions on how to use the modelset retrieval system.
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
    winsize_list = data["windowSize"]
    winsize_vars = ', '.join(['%s'] * len(winsize_list)) # Join single strings for each element in window size list for the SQL statement
    predhor_list = (data["predHor"])
    predhor_vars = ', '.join(['%s'] * len(predhor_list)) # Join single strings for each element in predHorList for the SQL statement
    perf_metric = (data["perfMetric"]) # Metric that should be used to calculate the performance score
    k = int(data["k"]) #Number of objects that should be returned to client
    weight = float(data["perfWeight"]) #Importance of Performance in comparison to Resource Awareness; Value [0-1]
    algorithm = data["algorithm"] #The algorithm that should be called. Possible values: fagin, threshold, naive.
   
   # Execute SQL Statement
    with connection:
        with connection.cursor() as cursor:
            if predhor_list == [] or predhor_list == 0: #SQL statement without filter for predHor
                cursor.execute(FILTER_MODELS_NO_PREDHOR.format(winsize_vars), (pID,) + tuple(winsize_list))
            else:cursor.execute(FILTER_MODELS.format(winsize_vars, predhor_vars), (pID,) + tuple(winsize_list + predhor_list))

            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'window_size', 'accuracy', 'mae', 'mse', 'rmse', '2'])
    
    if df.size == 0: #If no model match the requirements
        raise Exception ("No models found with specified metrics.")   

    df = get_perf_metric(df, perf_metric) #Get required Performance Metric 
    df = count_attributes(df) #Counts the number of attributes of each model

    if perf_metric == "acc":
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
            result = convert_to_json(fagin_topk(df_dict, weight, k), False, perf_metric)
        case 'threshold':
            result = convert_to_json(threshold_topk(df_dict, weight, k), False, perf_metric)
        case 'naive':
            result = convert_to_json(naive_topk(df, weight, k), False, perf_metric)
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
    winsize_list = data["windowSize"]
    winsize_vars = ', '.join(['%s'] * len(winsize_list)) # Join single strings for each element in winsize list for the SQL statement
    predhor_list = (data["predHor"])
    if len(predhor_list) == 1:
        raise Exception ("For modelset retrieval at least 2 prediction horizons must be selected!")
    predhor_vars = ', '.join(['%s'] * len(predhor_list)) # Join single strings for each element in predHorList for the SQL statement
    perf_metric = (data["perfMetric"]) # Metric that should be used to calculate the performance score
    k1 = int(data["k1"]) #Number of models that should be used per prediction Horizon to create modelsets
    if data["k2"] == "max": #Number of modelsets that should be returned to the client
        k2 = "max"
    else: k2 = int(data["n"])
    weight1 = float(data["perfWeight"]) #Importance of Performance in comparison to Resource Awareness; Value [0-1]
    weight2 = float(data["AMSWeight"]) #Importance of Modelset Score in comparison to Query Sharing Level; Value [0-1]
    algorithm = data["algorithm"] #The algorithm that should be called. Possible values: fagin, threshold, naive.
    combine_same_features = (data["combineSameFeatures"]) #Indicates, whether modelsets should only be generated if the models have the same features
    calculate_qsl = data["calculateQSL"]
    with connection:
        with connection.cursor() as cursor:
            if predhor_list == [] or predhor_list == 0:
                raise Exception ("One or multiple Prediction Horizons must be provided!")
            else:
                #Putting the variables into the statement
                cursor.execute(FILTER_MODELS_MODELSETS.format(winsize_vars, predhor_vars), (pID,) + tuple(winsize_list + predhor_list))
            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'prediction_horizon', 'window_size', 'accuracy', 'mae', 'mse', 'rmse', '2'])
    
    if df.size == 0: #If no model match the requirements
        raise Exception ("No models found with specified metrics.")   
    
    df = get_perf_metric(df, perf_metric) #get the required Performacne Metric
    df = count_attributes(df) #Counts the number of attributes of each model

    if perf_metric == "acc":
        df = normalize(df, '1', rev = False)
    else : # If error metric is chosen, reverse the normalization (Low error value is better)
        df = normalize(df, '1', rev = True)
    df = normalize(df, '2', rev = True)

    print(df.head)
    df_dict = {}
    df_dict_naive = {} #Dict of DF for the naive algorithm: No splitting done for performance/attributes

    #Splitting Table into smaller tables for each predHor value * 2 (performance and attributes)
    # df_dict with df_metric as value per predHor, which holds 2 more df 
    for key in predhor_list: 
        df_predhor = df.drop(df[df.prediction_horizon != int(key)].index)
        key = "predHor"+str(key)

        df_metric = {}
        df_perf = df_predhor.drop(columns=['2'])
        df_reaw = df_predhor.drop(columns=['1'])
        #df_reaw = df_predHor.drop(columns=['performance'])

        df_perf = df_perf.sort_values(by='1', ascending=False, na_position='first') #Sort with highest performance first
        df_reaw = df_reaw.sort_values(by='2', ascending=False, na_position='first') #Sort with least numner of attributes first

        df_metric.update([('1', df_perf), ('2', df_reaw)])
        df_dict.update({key: df_metric}) # Adding into dict each df containing a unique predHor
        df_dict_naive.update({key:df_predhor}) # Fill dict for naive algortithm

    print("First TopK:")
    # Call the desired algrotihm:
    result = []
    for key in df_dict:
        match algorithm:
            case 'fagin':
                result.append(fagin_topk(df_dict.get(key), weight1, k1))
            case 'threshold':
                #result = threshold_topk(df_dict, weight, k)
                result.append(threshold_topk(df_dict.get(key), weight1, k1))
            case 'naive':
                result.append(naive_topk(df_dict_naive.get(key), weight1, k1))
            case _:
                raise Exception ("Not a valid algorithm! Try 'naive', 'fagin', or 'threshold'.")

    combinations_raw = create_combinations(result, combine_same_features) # Create modelsets by combining models
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

    if k2 != "max": #If user put "max", all the created modelsets will be displayed.
        del combinations[k2:] #Delete every object from n to end of list
    else:
        k2 = len(combinations) # If max was set, return the total number of combinations to the client

    for index,modelset in enumerate(combinations): 
        modelset.update({'Modelset Number' : index+1 }) #Give each modelset a number
        qsl_list = []

        # Calculate Query Sharing Level
        cur_models = itertools.combinations(modelset['Models'].items(), 2) # Generate each combination of 2 models
        for model1, model2 in cur_models: 
            
            #model[1] is the model specs; split('-')[1] is the second part split by '-', so the features
            same_features =  model1[1]["model_name"].split('-')[1] == model2[1]["model_name"].split('-')[1] #Check if features are the same
            same_winsize =  model1[1]["window_size"] == model2[1]["window_size"] #Check if window size is the same

            if same_features and same_winsize: # Level 4: Same features
                qsl_list.append(4) 

            elif same_winsize: # Level 3: Same segmentation (= window size)
                qsl_list.append(3) 

            else:  # Level 0: No sharing possible
                qsl_list.append(0)

        # Aggregation function to determine QSL for entire modelset
        match calculate_qsl:
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
    df_from_od = pd.DataFrame(combinations) # Create DF from Ordered Dictionary
    df_from_od = df_from_od.rename(columns={"Modelset Score": "1", "Query Sharing Level": "2"})

    df_QSL = df_from_od.drop(columns=['1'])
    df_MSS = df_from_od.drop(columns=['2'])

    #print("Head", df_QSL.head)
    df_QSL = df_QSL.sort_values(by='2', ascending=False, na_position='first')
    df_MSS = df_MSS.sort_values(by='1', ascending=False, na_position='first')
    df_dict = {}
    df_dict.update([('QSL', df_QSL), ('MSS', df_MSS)])

    print("Second TopK:")
     # Call the desired algrotihm:
    match algorithm:
        case 'fagin':
            result = fagin_topk(df_dict, weight2, k2)
        case 'threshold':
            result = threshold_topk(df_dict, weight2, k2)
        case 'naive':
            result = naive_topk(df_from_od, weight2, k2)

    #print("Result", result)

    result_json= convert_to_json(result, True, perf_metric)
    return (result_json), 200