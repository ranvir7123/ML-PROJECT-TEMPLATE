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

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            para = param[model_name]
            
            print(f"Training {model_name}...")
            
            # Perform Grid Search CV
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            
            # Get the best model with tuned parameters
            best_model = gs.best_estimator_
            
            # Make predictions with the TUNED model
            y_test_pred = best_model.predict(X_test)
            
            # Calculate score
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            print(f"{model_name} score: {test_model_score:.4f}")

        return report
            
    except Exception as e:
        raise CustomException(e, sys)
    




def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)