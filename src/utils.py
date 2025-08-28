import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves a Python object to disk using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        print(f"Object successfully saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for model_name, model_object in models.items():
            model = model_object
            para = param[model_name]
            gs = GridSearchCV(model_object, para, cv=3) 
            gs.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
            
    except Exception as e:
        raise CustomException(e, sys)