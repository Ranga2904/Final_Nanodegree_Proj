import json
import numpy as np
import os
import joblib, pickle
from azureml.core import Model


def init():
    global daone
    model_path = Model.get_model_path('registered.sav')
    daone = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        data = data.reshape(1,-1)
        result = daone.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result
    except Exception as e:
        error = str(e)
        return error
