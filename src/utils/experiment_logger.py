# src/utils/experiment_logger.py
import os
import csv

class ExperimentLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Experiment ID", "Description", "R2", "MAE", "MSE"])

    def log(self, exp_id, description, r2, mae, mse):
        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([exp_id, description, r2, mae, mse])
