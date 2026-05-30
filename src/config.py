import yaml
from pydantic import BaseModel
from typing import Optional

class DataConfig(BaseModel):
    path: str
    test_size: float = 0.2
    random_state: int = 42

class ModelConfig(BaseModel):
    optimize_metric: str = "AUC"
    cv_folds: int = 10
    tune_iterations: int = 50
    feature_selection_threshold: float = 0.8

class LoggingConfig(BaseModel):
    level: str = "INFO"
    mlflow_experiment: str = "propensity_automl"

class PipelineConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
        return cls(**raw_config)