import os
import mlflow
import pandas as pd
from pycaret.classification import setup, compare_models, tune_model, finalize_model, save_model
from src.config import PipelineConfig
from src.logger import logger

def main():
    config = PipelineConfig.from_yaml("configs/pipeline.yaml")
    logger.info("Starting PyCaret AutoML Pipeline")

    df = pd.read_csv(config.data.path, sep=";")
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        logger.info("Leakage control: 'duration' column removed")
    
    df["y"] = df["y"].map({"no": 0, "yes": 1})
    logger.info("Data prepared", shape=df.shape)

    mlflow.set_tracking_uri("./mlruns")

    train_size = 1.0 - config.data.test_size
    
    setup(
        data=df,
        target="y",
        train_size=train_size,
        fold=config.model.cv_folds,
        fold_shuffle=True,
        fix_imbalance=True,
        fix_imbalance_method="smote",
        normalize=True,
        feature_selection=True,
        log_experiment=True,
        experiment_name=config.logging.mlflow_experiment,
        session_id=42,
        verbose=False
    )
    logger.info("PyCaret environment initialized")

    logger.info("Comparing models...")
    best_model = compare_models(sort="AUC", n_select=1, verbose=False)

    logger.info("Tuning best model...")
    tuned_model = tune_model(best_model, optimize="AUC", n_iter=config.model.tune_iterations, verbose=False)

    logger.info("Finalizing model on full dataset")
    final_model = finalize_model(tuned_model)

    os.makedirs("models", exist_ok=True)
    save_model(final_model, "models/final_propensity_model", verbose=False)
    
    logger.info("AutoML Pipeline completed successfully. Check MLflow UI for details.")

if __name__ == "__main__":
    main()