import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        print("Entered the predict pipeline")
        self.model_path = 'artifacts/model.pkl'
        self.preporcessor_path = 'artifacts/preprocessor.pkl'
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preporcessor_path)
    
    def predict(self, data: pd.DataFrame):
        try:
            data_scaled = self.preprocessor.transform(data)
            prediction = self.model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)
        return prediction
        
class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_frame(self):
        try:
            data = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
    
if __name__ == "__main__":
    data = CustomData("male", "group A", "some high school", "standard", "none", 70, 80)
    prediction_pipeline = PredictPipeline()
    prediction = prediction_pipeline.predict(data.get_data_as_frame())
