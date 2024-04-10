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
                WHERE parking_id = %s and period_minutes = %s and prediction_horizon = %s"""

FILTER_MODELS_NO_PREDHOR =  """SELECT model_id, model_name, accuracydt, accuracyrf, accuracylr, accuracyknn, attributes
                FROM niklas_trained_models
                WHERE parking_id = %s and period_minutes = %s"""

# TABLE_ATT = """SELECT model_id, model_name, attributes
#                 FROM niklas_trained_models
#                 WHERE parking_id = %s and period_minutes = %s"""