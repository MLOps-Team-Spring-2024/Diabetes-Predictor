import argparse

import pandas as pd

from mlops_team_project.src.model.model import model
from mlops_team_project.src.preprocess.preprocess import (
    train_test_split_and_write,
    min_max_scale_and_write,
)


def main() -> None:
    """
    Main function that runs the necessary steps for modeling.
    """
    parser = argparse.ArgumentParser(description="CLI for running XGBoost model")
    parser.add_argument("-t", "--test", type=str)

    args = parser.parse_args()

    df = pd.read_csv("data/raw/diabetes_data.csv")

    X_train, X_test, y_train, y_test = train_test_split_and_write(
        df=df, write_path="data/processed"
    )

    X_train_normalized, X_test_normalized = min_max_scale_and_write(
        X_train=X_train, X_test=X_test, write_path="data/processed"
    )

    model(
        X_train=X_train_normalized,
        X_test=X_test_normalized,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
