from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def train_test_split_and_write(
    df: pd.DataFrame,
    write_path: str,
    label_name: str = "Diabetes",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train and test sets and writes them to a folder"

    Args:
        df: Full dataset.
        write_path: The path where the split datasets will be written.
        label_name: The name of the column to use as labels.

    Returns:
        A tuple of dataframes in order of X_train, X_test, y_train, y_test.

    Raises:
        KeyError: If the label_name is not a column in the DataFrame.
    """
    try:
        labels_df = df[label_name]
        df.drop([label_name], axis=1, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            df, labels_df, test_size=test_size, random_state=17
        )

        return X_train, X_test, y_train, y_test
    except KeyError as e:
        raise KeyError(
            f"The label '{label_name}' is not a column in the DataFrame."
        ) from e


def min_max_scale_and_write(
    X_train: pd.DataFrame, X_test: pd.DataFrame, write_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales the training and testing data using Min-Max scaling and writes the normalized data.

    Args:
        X_train: Training data to scale.
        X_test: Testing data to scale.
        write_path (str, optional): The directory path where the scaled datasets will be written.

    Returns:
        Tuple with numpy arrays that have been min max scaled in order of X_train_normalized, X_test_normalized
    """
    min_max_scaler = MinMaxScaler().fit(X_train)
    X_train_normalized = min_max_scaler.transform(X_train)
    X_test_normalized = min_max_scaler.transform(X_test)

    return X_train_normalized, X_test_normalized
