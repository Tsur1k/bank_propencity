import pytest
import pandas as pd
from src.config import PipelineConfig

@pytest.fixture
def minimal_df():
    data = {
        "age": [30, 40, 50, 60, 70, 35, 45, 55, 65, 75],
        "job": ["admin"] * 10, "marital": ["single"] * 10, "education": ["secondary"] * 10,
        "default": ["no"] * 10, "housing": ["yes"] * 10, "loan": ["no"] * 10,
        "contact": ["cellular"] * 10, "day_of_week": ["mon"] * 10, "month": ["may"] * 10,
        "duration": [100, 200, 50, 300, 150, 120, 90, 110, 130, 140],
        "campaign": [1, 2, 1, 1, 3, 2, 1, 1, 2, 1],
        "pdays": [999, 999, -1, 999, 999, -1, 999, 999, 999, -1],
        "previous": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "poutcome": ["unknown"] * 10,
        "emp.var.rate": [1.1, 1.4, -1.8, 1.1, -2.0, 1.4, 1.1, 1.4, 1.1, -1.8],
        "cons.price.idx": [93.9, 93.1, 92.5, 93.9, 92.3, 93.1, 93.9, 93.1, 93.9, 92.5],
        "cons.conf.idx": [-42.0, -36.4, -41.8, -42.0, -46.2, -36.4, -42.0, -36.4, -42.0, -41.8],
        "euribor3m": [4.8, 4.9, 4.7, 4.8, 4.6, 4.9, 4.8, 4.9, 4.8, 4.7],
        "nr.employed": [5191, 5228, 5195, 5191, 5076, 5228, 5191, 5228, 5191, 5195],
        "y": ["no", "no", "yes", "no", "no", "yes", "no", "no", "no", "yes"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_config():
    return PipelineConfig.model_validate({
        "data": {"path": "dummy.csv", "test_size": 0.2, "random_state": 42},
        "model": {"optimize_metric": "AUC", "cv_folds": 3, "tune_iterations": 2, "feature_selection_threshold": 0.8},
        "logging": {"level": "WARNING", "mlflow_experiment": "test_exp"}
    })