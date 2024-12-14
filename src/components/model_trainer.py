import os
import sys
# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor

from xgboost import XGBRegressor
import warnings
from src.utils import save_object
from src.exception import CustomException
from dataclasses import dataclass
import logging
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Loading train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Train and test data loaded successfully")
            
            models = {
                "KNeighborsRegressor" : KNeighborsRegressor(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "AdaBoostRegressor"  : AdaBoostRegressor(),
                "SVR" : SVR(),
                "LinearRegression"  : LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                # "CatBoostRegressor" : CatBoostRegressor(),
                "XGBRegressor" : XGBRegressor()
            }
            model_report :dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                                models=models)
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            raise CustomException(e, sys)
