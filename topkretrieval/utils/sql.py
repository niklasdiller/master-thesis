SELECT_MODEL =  """SELECT model_id, model_name, developer, created_time, features, classifiers, model_size_in_bytes, 
                randomforestmaxdepth, kneighbours ,accuracy, mae, mse, rmse,
                FROM niklas_trained_models 
                WHERE model_name = %s
                ORDER BY model_id desc
                LIMIT %s;"""

SELECT_MODEL_DIRECT =  """SELECT model_id, model_name, developer, created_time, features, classifiers, model_size_in_bytes, 
                randomforestmaxdepth, kneighbours , accuracy, mae, mse, rmse
                FROM niklas_trained_models 
                WHERE model_name = %s
                ORDER BY model_id desc;"""

FILTER_MODELS =  """SELECT model_id, model_name, window_size, accuracy, mae, mse, rmse, features
                FROM niklas_trained_models
                WHERE parking_id = %s AND window_size IN ({}) AND prediction_horizon IN ({})"""

FILTER_MODELS_NO_PREDHOR =  """SELECT model_id, model_name, window_size, accuracy, mae, mse, rmse, features
                FROM niklas_trained_models
                WHERE parking_id = %s AND window_size IN ({})"""

FILTER_MODELS_MODELSETS = """SELECT model_id, model_name, prediction_horizon, window_size, accuracy, mae, mse, rmse, features
                FROM niklas_trained_models
                WHERE parking_id = %s AND window_size IN ({}) AND prediction_horizon IN ({})"""

# TABLE_ATT = """SELECT model_id, model_name, features
#                 FROM niklas_trained_models
#                 WHERE parking_id = %s and window_size = %s"""