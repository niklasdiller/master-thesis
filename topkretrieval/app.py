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
    weight = float(data["accWeight"]) #Importance of Accuracy in comparison to Resource Awareness; Value [0-1]
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
    df_dict = {}
    df_dict.update([('performance', df_perf), ('attributes', df_reaw)])

    #TODO: Determine what Algorithm should be called
    result = fagin_topk(df_dict, weight, k)
    #result = naive_topk(df, weight, k)

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
        result.append(df.iloc[ind])  # add to result list

    for ind in range(len(result)):
        round_result(result[ind]) # round values

    return convert_to_json(result)

def fagin_topk (df_dict: dict, weight, k: int):
    result = []
    i = 0
    seen = {} # TODO: Now a dict of dicts, Or rather a DF?
    all_seen = set()

    #Serial Access
    while True:
        for df in df_dict: #Look at every dataframe at same time
            df = df_dict.get(df)
            print(df.head)
            current = df.iloc[i] # Look at first item of dataframe
            id = str(current['model_id'])
            if id not in seen: #has not been seen before
                values = {}
                values.update({df.columns[2]:current.iloc[2]})
                # print(values)
                seen.update({id: values}) #Update the values
            else:  #has been seen before
                values = seen.get(id)
                values.update({df.columns[2]:current.iloc[2]}) #Put cell value for key of column name
                seen.update({id: values}) #Update the values
                if len(values) == 2:
                    all_seen.add(id)
        if len(all_seen) >= k:
            print("Seen all! ", all_seen, "Seen len: ", len(seen))
            break  
        i += 1
        if i == len(df.index)-1:
            return("Final row hit")
        #print("Row: ", i, " allseen: ", all_seen, " Total rows: ",len(df.index))
        

    #Random Access:
    for id in seen: # Iterate over all seen objects and fill in missing ones
        values = dict(seen.get(id))
        if len(values) != 2:
            keysList = list(values.keys())
            if keysList[0] == "performance": # If RA is missing
                df = df_dict.get('attributes') # Get correct df #TODO: Adjust for different RA metrics
                pd.DataFrame(df)
                val = df.loc[df['model_id'] == int(id), 'attributes'].values[0]
                values.update({'attributes': val})
                seen.update({id: values}) # Update the values
            else: # If Performance is missing
                df = df_dict.get('performance') #Get correct df
                pd.DataFrame(df)
                val = df.loc[df['model_id'] == int(id), 'performance'].values[0]
                values.update({'performance': val})
                seen.update({id: values}) # Update the values

    #print("Seen: " , seen)
    df_final = pd.DataFrame.from_dict(seen, orient='index') #Turning the dict into a df
    df_final.reset_index(inplace=True) #Reset index so model_id has own column
    df_final.columns = ['model_id', 'performance', 'attributes']
    #print(df_final.head)


    # Convert columns to numeric values
    df_final['performance'] = pd.to_numeric(df_final['performance'])
    df_final['attributes'] = pd.to_numeric(df_final['attributes'])
    
    for ind, row in df_final.iterrows(): # Calculating final score using weight
        score = (row['performance'] * weight) + (row['attributes'] * (1 - weight))
        df_final.at[ind, 'score'] = score

    df_final = df_final.sort_values(by='score', ascending=False, na_position='first') # sort by score

    print(df_final.head)

    for ind in range(k):
        result.append(df_final.iloc[ind])  # add to result list

    print(result)
    for ind in range(len(result)):
        round_result(result[ind]) # round values


    #TODO: Have model_name included in result to it can be converted into JSON
    #TODO: Have a look at warning messages in console by running topk()

    print (result)  
    #return convert_to_json(result)
    return ("Test")

