from dataclasses import dataclass

import pandas as pd
import pytest

from mlops_team_project.src.preprocess import train_test_split_and_write
from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

dataset_path = _PATH_DATA + "/processed"
df = pd.read_csv(_PATH_DATA + "/raw/diabetes_data.csv")

train_row = 56553
train_col = 17
test_row = 14139
test_col = 17
y_len = 56553


@pytest.fixture
def process_dataset():
    try:
        X_train, X_test, y_train, y_test = train_test_split_and_write(
            df=df, write_path="data/processed"
        )
        return [X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], len(y_train)]
    except TypeError:
        print("Cannot Unpack Non-iterable !")


def test_traintest_dataset(process_dataset):
    count = process_dataset
    assert count == [train_row, train_col, test_row, test_col, y_len]
