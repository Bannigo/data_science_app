import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        logging.info("Entered the data transformation method or component")
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Read the dataset as dataframe")

            logging.info(f"Missing values in train data:\n{train_df.isnull().sum()}")
            logging.info(f"Missing values in test data:\n{test_df.isnull().sum()}")

            # Define pipelines
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('std_scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
                ]
            )
            target_col = "math_score"
            # Split into features and target
            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            # Select numerical and categorical attributes
            num_attribs = list(input_feature_train_df.select_dtypes(include=[np.number]).columns)
            cat_attribs = list(input_feature_train_df.select_dtypes(include=['object', 'category']).columns)

            # Full preprocessing pipeline
            full_pipeline = ColumnTransformer([
                ('num', num_pipeline, num_attribs),
                ('cat', cat_pipeline, cat_attribs)
            ])

            # Apply transformations
            train_prepared = full_pipeline.fit_transform(input_feature_train_df)
            test_prepared = full_pipeline.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[
                train_prepared, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                test_prepared, np.array(target_feature_test_df)
            ]

            # Save the preprocessing pipeline
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=full_pipeline
            )
            logging.info("Data transformation is completed")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
