import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from src.models.autoencoder import AutoencoderFeatureExtractor
from sklearn.model_selection import KFold
import numpy as np



class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)


    def load_data(self):
            dataset_path_map = {
                "landsat_p": "data/processed/1_prf2000_mean_raw_Aug_9input_PMinMax.csv",
                "landsat_q": "data/processed/2_prf2000_mean_raw_Aug_9input_QMinMax.csv",  # Already used
                "lidar_p": "data/processed/3_tvol_lidarLivePMinMax.csv",
                "lidar_q": "data/processed/4_tvol_lidarLiveQMinMax.csv",
                "lidar_q_top_selected_features" : "data/processed/lidar_q_top20_modelselect.csv"
                }
            path = dataset_path_map.get(self.config["dataset"])
            if path is None:
                raise ValueError("Invalid dataset selected.")
            
            df = pd.read_csv(path)
            df = df.drop(columns=[col for col in ['origin', 'Origin', 'status', 'Status'] if col in df.columns])
            return df


    def run_pipeline(self):
        from src.utils.experiment_logger import ExperimentLogger
        logger = ExperimentLogger("outputs/experiment_results.csv")

        df = self.load_data()
        X = df.drop(columns=["Target"])
        y = df["Target"]

        kf = KFold(n_splits=5, shuffle=True, random_state=self.config["random_seed"])
        r2_scores, mae_scores, mse_scores = [], [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # --- Autoencoder Only ---
            if self.config["feature_selection"] == "autoencoder":
                print("Training autoencoder...")
                ae = AutoencoderFeatureExtractor(input_dim=X_train.shape[1], latent_dim=8)
                ae.train(X_train)

                print("Extracting deep features...")
                X_train = ae.extract_features(X_train)
                X_test = ae.extract_features(X_test)

            # --- Autoencoder + Raw Fusion ---
            if self.config["feature_selection"] == "autoencoder+raw":
                print("Training autoencoder...")
                ae = AutoencoderFeatureExtractor(input_dim=X_train.shape[1], latent_dim=16)
                ae.train(X_train)

                print("Extracting deep features...")
                X_train_latent = ae.extract_features(X_train)
                X_test_latent = ae.extract_features(X_test)
                X_train = np.concatenate([X_train.values, X_train_latent], axis=1)
                X_test = np.concatenate([X_test.values, X_test_latent], axis=1)

            # --- RFE ---
            if self.config["feature_selection"] == "rfe":
                selector = RFE(RandomForestRegressor(n_estimators=10), n_features_to_select=20)
                selector.fit(X_train, y_train)
                selected_features = X_train.columns[selector.get_support()]
                print("Selected features by RFE:")
                print(selected_features.tolist())
                print(f"Training model on shape: {X_train.shape}")

                #  Now transform
                X_train = selector.transform(X_train)
                X_test = selector.transform(X_test)

            # --- Model Setup ---
            if self.config["model_type"] == "random_forest":
                model = RandomForestRegressor(n_estimators=self.config["n_estimators"])
            elif self.config["model_type"] == "xgboost":
                model = XGBRegressor(n_estimators=self.config["n_estimators"], random_state=self.config["random_seed"])
            else:
                raise ValueError("Model type not implemented yet.")

            # --- Train & Evaluate ---
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_scores.append(r2_score(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            mse_scores.append(mean_squared_error(y_test, y_pred))

        print("Average R²:", np.mean(r2_scores))
        print("Average MAE:", np.mean(mae_scores))
        print("Average MSE:", np.mean(mse_scores))

        logger.log(
            exp_id="ExpLiQ",
            description="XGBoost on raw + SAE latent features",
            r2=np.mean(r2_scores),
            mae=np.mean(mae_scores),
            mse=np.mean(mse_scores)
    )
