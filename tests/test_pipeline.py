import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import src.train

@patch("src.train.predict_model")
@patch("src.train.mlflow")
@patch("src.train.save_model")
@patch("src.train.finalize_model")
@patch("src.train.tune_model")
@patch("src.train.compare_models")
@patch("src.train.setup")
@patch("pandas.read_csv")
def test_pycaret_pipeline_orchestration(mock_read_csv, mock_setup, mock_compare, mock_tune, mock_finalize, mock_save, mock_mlflow, mock_predict, minimal_df):
    mock_read_csv.return_value = minimal_df
    mock_setup.return_value = {"test": pd.DataFrame()}
    mock_compare.return_value = "best_model"
    mock_tune.return_value = "tuned_model"
    mock_finalize.return_value = "final_model"
    mock_predict.return_value = {"y_pred": [0, 1]}

    mock_mlflow_ctx = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = mock_mlflow_ctx
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    src.train.main()

    mock_read_csv.assert_called_once()
    mock_setup.assert_called_once()
    mock_compare.assert_called_once()
    mock_tune.assert_called_once()
    mock_finalize.assert_called_once()
    mock_save.assert_called_once()
    
    assert mock_mlflow.set_experiment.called
    assert mock_mlflow.start_run.called