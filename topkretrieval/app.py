from email import utils
from tokenize import String
from flask import Flask, request, jsonify
import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn' # Ignores warnings regarding chaning values to df copy

from utils.helper import *
from utils.sql import *
from threshold import *

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
    weight = float(data["accWeight"]) #Importance of Performance in comparison to Resource Awareness; Value [0-1]
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(FILTER_MODELS, (pID, perMin))
            df = pd.DataFrame(cursor.fetchall(), columns=['model_id', 'model_name', 'accuracydt', 'accuracyrf', 'accuracylr', 'accuracyknn', 'attributes'])
            
    df = reshape_perf_table(df) #Call function that summarizes the accuracy for acc
    df = count_attributes(df) #Call function that counts the number of attributes of each model

    df = normalize(df, 'performance', rev = False)
    df = normalize(df, 'attributes', rev = True)

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

    list = [df_perf, df_reaw]
    print(threshold_topk(list, weight, k))
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

def fagin_topk (df_dict: dict, weight, k: int): #TODO: Improve efficiency by getting rif of multiple for loops
    result = []
    i = 0
    seen = {} # TODO: Now a dict of dicts converted to a DF later, Or rather a DF directly ?
    all_seen = set()

    # Step 1: Serial Access
    while True:
        for df in df_dict: #Look at every dataframe at same time
            df = df_dict.get(df)
            #print(df.head)
            current = df.iloc[i] # Look at first item of dataframe
            id = str(current['model_id'])
            if id not in seen: #has not been seen before
                values = {}
                values.update({df.columns[1]:current.iloc[1], df.columns[2]:current.iloc[2]}) #1: model_name, 2: Metric
                # print(values)
                seen.update({id: values}) #Update the values
            else:  #has been seen before
                values = seen.get(id)
                values.update({df.columns[2]:current.iloc[2]}) #Put cell value for key of column name
                seen.update({id: values}) #Update the values
                if len(values) == 3: #Name, Performance and RA = 3
                    all_seen.add(id)
        if len(all_seen) >= k:
            print("Seen all! ", all_seen, "Seen len: ", len(seen))
            break  
        i += 1
        if i == len(df.index)-1:
            return("Final row hit")
        #print("Row: ", i, " allseen: ", all_seen, " Total rows: ",len(df.index))
        

    # Step 2: Random Access
    for id in seen: # Iterate over all seen objects and fill in missing ones
        values = dict(seen.get(id))
        if len(values) != 3:
            keysList = list(values.keys())
            try:
                if keysList[1] == "performance": # If RA is missing
                    df = df_dict.get('attributes') # Get correct df #TODO: Adjust for different/more RA metrics
                    pd.DataFrame(df)
                    val = df.loc[df['model_id'] == int(id), 'attributes'].values[0]
                    values.update({'attributes': val})
                    seen.update({id: values}) # Update the values
                elif keysList[1] == "attributes": # If Performance is missing
                    df = df_dict.get('performance') #Get correct df
                    pd.DataFrame(df)
                    val = df.loc[df['model_id'] == int(id), 'performance'].values[0]
                    values.update({'performance': val})
                    seen.update({id: values}) # Update the values
            except Exception as e: print("Model seems to be missing one or more relevant metrics.")

    # Turning the dict into a df            
    df_final = pd.DataFrame.from_dict(seen, orient='index') 
    df_final.reset_index(inplace=True) #Reset index so model_id has own column
    df_final.columns = ['model_id', 'model_name', 'performance', 'attributes']

    # Convert columns to numeric values
    df_final['performance'] = pd.to_numeric(df_final['performance'])
    df_final['attributes'] = pd.to_numeric(df_final['attributes'])


    # Step 3: Computing the grade
    for ind, row in df_final.iterrows(): 
        score = (row['performance'] * weight) + (row['attributes'] * (1 - weight))
        df_final.at[ind, 'score'] = score

    df_final = df_final.sort_values(by='score', ascending=False, na_position='first') # sort by score

    for ind in range(k):
        result.append(df_final.iloc[ind])  # add to result list

    print("Result:" , result)
    for ind in range(len(result)):
        round_result(result[ind]) # round values

    #print (result)  
    return convert_to_json(result)


def threshold_topk (df_list, weight, k):
    i = 0 
    result = pd.DataFrame(columns =['model_id', 'model_name', 'score'] )
    while True:
        threshold = 0

        for cur_df_index, cur_df in enumerate(df_list): # Keep track of index to delete cur_df later
            #print("Current DF: ", cur_df.head())
            other_dfs = df_list.copy()
            del other_dfs[cur_df_index] # Establish a list that has every list but the current one to look into

            # print("Cur DF: ", cur_df)
            # print("Other DFs: ", other_dfs)

            cur_row = cur_df.iloc[i]
            cur_id = cur_row['model_id']

            # Calculate Threshold:
            if cur_row.index[-1] == 'performance':
                threshold += cur_row.iloc[-1]*weight
            elif cur_row.index[-1] == 'attributes':
                threshold += cur_row.iloc[-1]*(1-weight)

            # Random Access:
            for other_df in other_dfs:
                metric = other_df.columns[-1]
                other_val = other_df.loc[other_df['model_id'] == cur_id, metric].values[0]
                if cur_id == '2050':
                    print("Gefunden!!!!")
                if metric == 'performance': #performance is missing
                   # print("Val last col: ",  cur_row.iloc[-1] )
                    cur_row.iloc[-1] = cur_row.iloc[-1]*(1-weight) + other_val*weight
                    #print("Val last col after calc: ",  cur_row.iloc[-1] )
                elif metric == 'attributes': #RA is missing
                    cur_row.iloc[-1] = cur_row.iloc[-1]*weight + other_val*(1-weight)

                else: #TODO: Handle multiple metrics
                    raise Exception ("Unknown metric in data!")

                # Adding seen model to result:
                if cur_id not in result['model_id'].values: #Ignore if model already in result
                    cur_row = cur_row.rename({cur_row.index[-1] : 'score'}) #rename attributes/performance to score
                    #print("Result before adding: ", result)
                    result.loc[len(result)] = cur_row #Add complete seen item to end of result df

                    #print("Result after adding: ", result)
                    result = result.sort_values(by='score',ascending=False) #Sort values so worst can be dropped
                    result = result.reset_index(drop = True) #Reset Index 
                    #print("Result after sorting: ", result)

                    # Only keep k best models:
                    if result.shape[0] > k:
                        #print ("Result too long! Length: ", result.shape[0])
                        result = result.drop(result.tail(1).index) #drop last row if result is longer than k
                        #print("Now length: ", result.shape[0])
                else: 
                    print("Model already in result")
        
        #print ("threshold: ", threshold)
        #print("Result: ", result)
        #print(result.shape[0] == k)
        #print(result.iloc[-1, -1] >= threshold)

        # Stop condition:
        if result.shape[0] == k and result.iloc[-1, -1] >= threshold:
            print("Final Result: ", result)
            return result
        
        # if i == 100:
        #     print ("OOB")
        #     return None
        #print("Incrementing i")
        
        i += 1    
