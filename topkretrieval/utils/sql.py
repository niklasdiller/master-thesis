SELECT_MODEL =  """SELECT model_id, model_name, developer, created_time, attributes, classifiers, model_size_in_bytes, 
                randomforestmaxdepth, kneighbours ,accuracy, mae, mse, rmse,
                FROM niklas_trained_models 
                WHERE model_name = %s
                ORDER BY model_id desc
                LIMIT %s;"""

SELECT_MODEL_DIRECT =  """SELECT model_id, model_name, developer, created_time, attributes, classifiers, model_size_in_bytes, 
                randomforestmaxdepth, kneighbours , accuracy, mae, mse, rmse
                FROM niklas_trained_models 
                WHERE model_name = %s
                ORDER BY model_id desc;"""

FILTER_MODELS =  """SELECT model_id, model_name, period_minutes, accuracy, mae, mse, rmse, attributes
                FROM niklas_trained_models
                WHERE parking_id = %s AND period_minutes IN ({}) AND prediction_horizon IN ({})"""

FILTER_MODELS_NO_PREDHOR =  """SELECT model_id, model_name, period_minutes, accuracy, mae, mse, rmse, attributes
                FROM niklas_trained_models
                WHERE parking_id = %s AND period_minutes IN ({})"""

FILTER_MODELS_MODELSETS = """SELECT model_id, model_name, prediction_horizon, period_minutes, accuracy, mae, mse, rmse, attributes
                FROM niklas_trained_models
                WHERE parking_id = %s AND period_minutes IN ({}) AND prediction_horizon IN ({})"""

# TABLE_ATT = """SELECT model_id, model_name, attributes
#                 FROM niklas_trained_models
#                 WHERE parking_id = %s and period_minutes = %s"""