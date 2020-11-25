import json
import numpy as np
import os
import joblib, pickle
from azureml.core import Model


def init():
    model_path = Model.get_model_path('registered.sav')
    model = joblib.load(model_path)

def run(data):
    try:
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
