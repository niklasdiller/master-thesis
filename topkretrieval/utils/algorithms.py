import pandas as pd
from utils.helper import *
from utils.sql import *
pd.options.mode.chained_assignment = None  # default='warn' # Ignores warnings regarding chaning values to df copy


def naive_topk (df: pd.DataFrame, weight: float, k: int):
    result = []
    print(df)
    for ind in df.index:
       #Compute Score and put in new column
       score = (df.at[ind, 'performance'] * weight) + (df.at[ind, 'attributes'] * (1-weight)) # Compute score
       df.at[ind, 'score'] = score

    df = df.sort_values(by='score', ascending=False, na_position='first')  #Sort for score

    for ind in range(k):
        result.append(df.iloc[ind])  # add k models to result list

    for ind in range(len(result)):
        round_result(result[ind]) # round values

    return result


# def faginver2 (df_list, weight, k):
#     result = pd.DataFrame(columns =['model_id', 'model_name', 'performance', 'attributes', 'score'] )
#     i = 0
#     seen = {}

#     while True:
#         for cur_df_index, cur_df in enumerate(df_list):
#             cur_row = cur_df.iloc[i]
#             cur_id = cur_row['model_id']
#             if cur_id not in seen:
#                 seen[cur_id] = (cur_row['model_name'], cur_row['performance'], cur_row['attributes'], cur )
#                 print(seen)

def fagin_topk (df_dict: dict, weight, k: int): #TODO: Improve efficiency by getting rif of multiple for loops
    result = []
    i = 0
    seen = {} # TODO: Now a dict of dicts converted to a DF later, Or rather a DF directly ?
    all_seen = 0

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
                    all_seen += 1
        if all_seen >= k:
            #print("Seen all! ", all_seen, "Seen len: ", len(seen))
            break  
        i += 1
        if i == len(df.index)-1:
            raise Exception("Final row hit.")
        #print("Row: ", i, " allseen: ", all_seen, " Total rows: ",len(df.index))
        

    # Step 2: Random Access
    for id in seen: # Iterate over all seen objects and fill in missing ones
        values = dict(seen.get(id))
        if len(values) != 3:
            keysList = list(values.keys())
            try:
                if keysList[1] == 'performance': # If RA is missing
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
    df_final['score'] = (df_final['performance'] * weight) + (df_final['attributes'] * (1 - weight))

    df_final = df_final.sort_values(by='score', ascending=False, na_position='first') # sort by score

    for ind in range(k):
        result.append(df_final.iloc[ind])  # add to result list

    #print("Result:" , result)
    for ind in range(len(result)):
        round_result(result[ind]) # round values

    return result



def threshold_topk (df_dict: dict, weight: float, k: int):
    i = 0 
    result_df = pd.DataFrame(columns =['model_id', 'model_name', 'performance', 'attributes', 'score'] )
    result = []
    while True:
        threshold = 0

        for key in df_dict: # Keep track of index to delete cur_df later
            cur_df = df_dict.get(key)
            other_dfs = df_dict.copy()
            del other_dfs[key] # Establish a list that has every list but the current one to look into

        #     # print("Cur DF: ", cur_df)
        #     # print("Other DFs: ", other_dfs)

            cur_row = cur_df.iloc[i]
            cur_id = cur_row['model_id']

            # Calculate Threshold:
            if cur_row.index[-1] == 'performance':
                threshold += cur_row.iloc[-1]*weight
            elif cur_row.index[-1] == 'attributes':
                threshold += cur_row.iloc[-1]*(1-weight)

            # Random Access:
            for key in other_dfs:
                other_df = other_dfs[key]
                other_metric = other_df.columns[-1] # Last column
                other_val = other_df.loc[other_df['model_id'] == cur_id, other_metric].values[0] # Look for model id in other df and get missing value
                cur_val = cur_row.iloc[-1] #Last metric

                if other_metric == 'performance': #performance is missing
                   # print("Val last col: ",  cur_row.iloc[-1] )
                    cur_row['score'] = cur_val*(1-weight) + other_val*weight #calculating score
                    cur_row['performance'] = other_val #filling in the missing metric
                    #print("Val last col after calc: ",  cur_row.iloc[-1] )
                elif other_metric == 'attributes': #RA is missing
                    cur_row['score'] = cur_val*weight + other_val*(1-weight)
                    cur_row['attributes'] = other_val

                else: #TODO: Handle multiple metrics
                    raise Exception ("Unknown metric in data!")

                # Adding seen model to result:
                if cur_id not in result_df['model_id'].values: #Ignore if model already in result
                    #print("Result before adding: ", result)
                    result_df.loc[len(result_df)] = cur_row #Add complete seen item to end of result df

                    #print("Result after adding: ", result)
                    result_df = result_df.sort_values(by='score',ascending=False) #Sort values so worst model can be dropped
                    result_df = result_df.reset_index(drop = True) #Reset Index 
                    #print("Result after sorting: ", result)

                    # Only keep k best models:
                    if result_df.shape[0] > k:
                        #print ("Result too long! Length: ", result.shape[0])
                        result_df = result_df.drop(result_df.tail(1).index) #drop last row if result is longer than k
                        #print("Now length: ", result.shape[0])
                else: 
                    print("Model already in result")

        # Stop condition:
        if result_df.shape[0] == k and result_df.iloc[-1, -1] >= threshold:
            result_df = round_result_df(result_df) #Round the values of the DF
            for ind in range(k):
                result.append(result_df.iloc[ind])  # add to result list
            #result_df = result_df.to_json(orient='index') #Convert to JSON
            #print("Final Result: ", result)
            return result

        i += 1    