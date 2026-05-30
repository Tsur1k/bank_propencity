import pytest
import pandas as pd
from tests.conftest import minimal_df

def test_leakage_control(minimal_df):
    df = minimal_df.copy()
    assert "duration" in df.columns
    df = df.drop(columns=["duration"])
    assert "duration" not in df.columns

def test_target_encoding(minimal_df):
    df = minimal_df.copy()
    df["y"] = df["y"].map({"no": 0, "yes": 1})
    assert set(df["y"].unique()) == {0, 1}
    assert df["y"].dtype in ["int64", "int32"]