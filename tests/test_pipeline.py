from unittest.mock import patch
import src.train

@patch("src.train.save_model")
@patch("src.train.finalize_model")
@patch("src.train.tune_model")
@patch("src.train.compare_models")
@patch("src.train.setup")
@patch("pandas.read_csv")
def test_pycaret_pipeline_orchestration(mock_read_csv, mock_setup, mock_compare, mock_tune, mock_finalize, mock_save, minimal_df):
    mock_read_csv.return_value = minimal_df
    mock_setup.return_value = None
    mock_compare.return_value = "best_model"
    mock_tune.return_value = "tuned_model"
    mock_finalize.return_value = "final_model"

    src.train.main()

    mock_read_csv.assert_called_once()
    mock_setup.assert_called_once()
    mock_compare.assert_called_once()
    mock_tune.assert_called_once()
    mock_finalize.assert_called_once()
    mock_save.assert_called_once()