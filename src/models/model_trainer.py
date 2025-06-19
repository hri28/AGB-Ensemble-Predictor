import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE


class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)


    def load_data(self):
            dataset_path_map = {
                "landsat": "data/processed/landsat.csv",
                "lidar": "data/processed/lidar.csv",
                "fusion": "data/processed/fusion.csv",
                "poly_landsat": "data/processed/prf2000_mean_raw_Aug_9input_QMinMax.csv"
            }
            path = dataset_path_map.get(self.config["dataset"])
            if path is None:
                raise ValueError("Invalid dataset selected.")
            return pd.read_csv(path)


    def run_pipeline(self):
        from src.utils.experiment_logger import ExperimentLogger

        logger = ExperimentLogger("outputs/experiment_results.csv")
        #load and split train and test data
        df = self.load_data()
        X = df.drop(columns=["Target"])
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config["test_size"], random_state=self.config["random_seed"])

        if self.config["feature_selection"] == "rfe":
            selector = RFE(RandomForestRegressor(n_estimators=50), n_features_to_select=20)
            selector.fit(X_train, y_train)
            
            #print feature names
            selected_features = X.columns[selector.get_support()]
            print("Selected features by RFE:")
            print(selected_features.tolist())


            # Reduce both train and test sets
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)

        #initialize the model
        if self.config["model_type"] == "random_forest":
            model = RandomForestRegressor(n_estimators=self.config["n_estimators"])
        elif self.config["model_type"] == "xgboost":
            model = XGBRegressor(n_estimators=self.config["n_estimators"], random_state=self.config["random_seed"])
        else:
            raise ValueError("Model type not implemented yet.")
        
        #train model
        model.fit(X_train, y_train)

        #evaluate predictions
        y_pred = model.predict(X_test)

        print("R²:", r2_score(y_test, y_pred))
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))
        logger.log(
            exp_id="Exp03",
            description="Random Forest with RFE (top 20 features) on polynomial Landsat",
            r2=r2_score(y_test, y_pred),
            mae=mean_absolute_error(y_test, y_pred),
            mse=mean_squared_error(y_test, y_pred)
)
