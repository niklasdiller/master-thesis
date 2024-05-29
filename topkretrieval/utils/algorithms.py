import pandas as pd
import time
from utils.helper import *
from utils.sql import *
pd.options.mode.chained_assignment = None  # default='warn' # Ignores warnings regarding chaning values to df copy


def naive_topk (df: pd.DataFrame, weight: float, k: int):
    #print("Head", df.head)
    print("df.size", df.size)
    result = []
    for ind in df.index:
        #Compute Score and put in new column
        score = (df.at[ind, '1'] * weight) + (df.at[ind, '2'] * (1-weight)) # Using the last 2 columns (= Performance/Attributes or MSS/QSL)
        #score = (df.at[ind, df.columns[-2]] * weight) + (df.at[ind, df.columns[-1]] * (1-weight))

        df.at[ind, 'score'] = score

    df = df.sort_values(by='score', ascending=False, na_position='first')  #Sort for score

    for ind in range(k):
        result.append(df.iloc[ind])  # add k models to result list

    for ind in range(len(result)):
        round_result(result[ind]) # round values

    return result


def fagin_topk (df_dict, weight, k: int):
    #print("DF dict anfang", df_dict)
    result = []
    i = 0
    seen = {} 
    all_seen = 0
    ran_acc = 0

    # Step 1: Serial Access
    while True:
        for df in df_dict: #Look at every dataframe at same time
            df = df_dict.get(df)
            current = df.iloc[i] # Look at first item of dataframe
            id = current.iloc[0] # The ModelID/ModelsetID 
            if id not in seen: #has not been seen before
                values = {}
                # print("Current", current)
                # print("current.iloc", current[1])
                for col_name in df.columns:  # For each column:
                    values[col_name] = current[col_name]  # Fill values with that column
                #print("Values", values)
                #values.update({df.columns[1]:current.iloc[1], df.columns[-1]:current.iloc[-1]}) #1: model_name, -1: Metric
                seen.update({id: values}) #Update the values
            else:  #has been seen before
                values = seen.get(id)
                values.update({df.columns[-1]:current.iloc[-1]}) #Put cell value for key of column name
                seen.update({id: values}) #Update the values
                if values["1"] != None and values["2"] != None: #Both metrics have been seen
                    all_seen += 1
        if all_seen >= k:
            print("Seen all! All seen length:", all_seen, " Seen length: ", len(seen), "Number of sorted accesses: ", i)
            break  
        i += 1
        if i == len(df.index):
            print("No more . No more models/modelsets to inspect. All seen length:", all_seen)
            raise Exception("Final row hit. Chosen n is probably too high. Try setting n = 'max'")
        #print("Row: ", i, " allseen: ", all_seen, " Total rows: ",len(df.index))
        
    # Step 2: Random Access
    for id in seen: # Iterate over all seen objects and fill in missing ones
        values = dict(seen.get(id))
        if "1" not in values or "2" not in values: #If one metric is still missing
            keysList = list(values.keys())
            #try:
            if keysList[-1] == '1': # If RA is missing
                df = df_dict.get('2') # Get missing df
                pd.DataFrame(df)
                val = df.loc[df.iloc[:, 0] == int(id), '2'].values[0] #Where the first column (=id) is the considered id add the missing metric
                values.update({'2': val})
                seen.update({id: values}) # Update the values
                ran_acc += 1
            elif keysList[-1] == "2": # If Performance is missing
                df = df_dict.get('1') #Get missing df
                pd.DataFrame(df)
                val = df.loc[df.iloc[:, 0] == int(id), '1'].values[0] #Where the first column (=id) is the considered id add the missing metric
                values.update({'1': val})
                seen.update({id: values}) # Update the values
                ran_acc += 1
            #except Exception as e: print("Model seems to be missing one or more relevant metrics.")

    # Turning the dict into a df            
    df_final = pd.DataFrame(columns =['model_id', 'model_name', 'perfMetric', 'prediction_horizon', 'window_size', 'Models', '1', '2', 'score'] )
    df_final = pd.DataFrame.from_dict(seen, orient='index')

    # Convert columns to numeric values
    df_final['1'] = pd.to_numeric(df_final['1'])
    df_final['2'] = pd.to_numeric(df_final['2'])

    # Step 3: Computing the grade 
    df_final['score'] = (df_final['1'] * weight) + (df_final['2'] * (1 - weight))

    df_final = df_final.sort_values(by='score', ascending=False, na_position='first') # sort by score

    for ind in range(k):
        result.append(df_final.iloc[ind])  # add to result list

    #print("Result:" , result)
    for ind in range(len(result)):
        round_result(result[ind]) # round values

    print("Number of random accesses: ", ran_acc)
    return result


def threshold_topk (df_dict, weight: float, k: int):
    #print("DF dict am Anfang: ", df_dict)
    i = 0 
    result = []
    ran_acc = 0
    while True:
        threshold = 0
        for key in df_dict: # Keep track of index to delete cur_df later
            cur_df = df_dict.get(key)
            other_dfs = df_dict.copy()
            del other_dfs[key] # Establish a list that has every list but the current one to look into
            if len(other_dfs) != 1: # Errorhandling: More metrics than perf and RA are listed
               raise Exception ("Looking at more than 2 metrics!")
            # print("Cur DF: ", cur_df)
            # print("Other DFs: ", other_dfs)

            cur_row = cur_df.iloc[i]
            cur_id = cur_row.iloc[0] # The ModelID/ModelsetID 

            # Calculate Threshold:
            if cur_row.index[-1] == '1':
                threshold += cur_row.iloc[-1]*weight
            elif cur_row.index[-1] == '2':
                threshold += cur_row.iloc[-1]*(1-weight)

            # Random Access:
            if cur_id not in [row.iloc[0] for row in result]: #If the current object has not been looked at before
                for other_key, other_df in other_dfs.items(): # is only one key
                    other_row = other_df.loc[other_df.iloc[:, 0] == cur_id] # getting row for same object in other list
                    other_metric = other_df.columns[-1] # Last column
                    other_val = other_row.iloc[0][other_metric] # Getting value of missing metric

                if other_metric == '1': #performance is missing
                    cur_row['score'] = cur_row.iloc[-1]*(1-weight) + other_val*weight #calculating score
                    cur_row['1'] = other_val #filling in the missing metric
                    ran_acc += 1
                elif other_metric == '2': #RA is missing
                    cur_row['score'] = cur_row.iloc[-1]*weight + other_val*(1-weight) #calculating score
                    cur_row['2'] = other_val #filling in the missing metric
                    ran_acc += 1
                else:
                    print(other_df.head)
                    raise Exception ("Unknown metric in data!")

                # Adding seen model to result:
                result.append(cur_row)
                #print("Result after adding: ", result)
                result = sorted(result, key=lambda x: ['score'], reverse=True) #Sort values so worst model can be dropped
                #print("Result after sorting: ", result)

                # Only keep k best models:
                if len(result) > k:
                    #print ("Result too long! Length: ", len(result))
                    result= result[:k]
                    #print("Now length: ", len(result))

        # Stop condition:
        if len(result) >= k and result[-1]['score'] >= threshold:
            time_stop1 = time.time()
            #print("Final Result: ", result)
            for ind in range(len(result)):
                round_result(result[ind]) # round values
            print("Number of sorted accesses: ", i, "Number of random accesses: ", ran_acc)

            return result

        i += 1 