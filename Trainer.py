import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score  # Accuracy metrics
import pickle

import config


class Trainer:

    def __init__(self):
        data = pd.read_csv(config.BODY_POINTS_FILENAME, header=None)
        data.fillna(0, inplace=True)
        gestures = data.iloc[:, 0]
        points = data.drop(columns=[0])
        self.points_train, self.points_test, self.gestures_train, self.gestures_test = train_test_split(
            points, gestures, test_size=0.3, random_state=1234)
        self.pipelines = {
            'lr': make_pipeline(StandardScaler(), LogisticRegression()),
            'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }

    def fit_models(self):
        fit_models = {}
        for algo, pipeline in self.pipelines.items():
            model = pipeline.fit(self.points_train, self.gestures_train)
            fit_models[algo] = model
        return fit_models

    def save_models(self, models):
        for algo, model in models.items():
            yhat = model.predict(self.points_test)
            print(algo, accuracy_score(self.gestures_test, yhat))
            with open(f'{config.FITED_MODELS_DIR}\\{algo}.pkl', 'wb') as f:
                pickle.dump(model, f)

    def train_and_save_model(self):
        models = self.fit_models()
        self.save_models(models)

    @staticmethod
    def load_model(model_name=""):
        if model_name == "":
            model_name = config.SELECTED_MODEL
        with open(f'{config.FITED_MODELS_DIR}\\{model_name}.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def predict(all_body_points, model):
        bd_points_pd = pd.DataFrame([all_body_points])
        body_language_class = model.predict(bd_points_pd)[0]
        body_language_prob = model.predict_proba(bd_points_pd)[0]
        return body_language_class, body_language_prob


#Trainer().train_and_save_model()
