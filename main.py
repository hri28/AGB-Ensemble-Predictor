from src.models.model_trainer import ModelTrainer

if __name__ == "__main__":
    trainer = ModelTrainer("config.json")
    trainer.run_pipeline()
