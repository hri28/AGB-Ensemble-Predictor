import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def load_data(self):
        if self.config["dataset"] == "landsat":
            return pd.read_csv("data/processed/landsat.csv")
        elif self.config["dataset"] == "lidar":
            return pd.read_csv("data/processed/lidar.csv")
        elif self.config["dataset"] == "fusion":
            return pd.read_csv("data/processed/fusion.csv")
        else:
            raise ValueError("Invalid dataset selected.")

    def run_pipeline(self):
        df = self.load_data()
        X = df.drop(columns=["Target"])
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config["test_size"], random_state=self.config["random_seed"])

        if self.config["model_type"] == "random_forest":
            model = RandomForestRegressor(n_estimators=self.config["n_estimators"])
        else:
            raise ValueError("Model type not implemented yet.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("R²:", r2_score(y_test, y_pred))
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))
