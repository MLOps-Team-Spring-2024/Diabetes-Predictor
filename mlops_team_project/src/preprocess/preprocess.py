from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def train_test_split_and_write(
    df: pd.DataFrame,
    label_name: str = "Diabetes",
    write_path: str = "output",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train and test sets and writes them to a folder"

    Parameters:
    - df (pd.DataFrame): Full dataset.
    - label_name (str, optional): The name of the column to use as labels.
    - write_path (str, optional): The path where the split datasets will be written.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing training features,
      testing features, training labels, and testing labels.

    Raises:
    - KeyError: If the label_name is not a column in the DataFrame.
    """
    try:
        labels_df = df[label_name]
        df.drop([label_name], axis=1, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            df, labels_df, test_size=test_size, random_state=17
        )

        X_train.to_csv(f"{write_path}/train_data.csv", index=False)
        X_test.to_csv(f"{write_path}/test_data.csv", index=False)
        y_train.to_csv(f"{write_path}/train_labels.csv", index=False)
        y_test.to_csv(f"{write_path}/test_labels.csv", index=False)

        return X_train, X_test, y_train, y_test
    except KeyError as e:
        raise KeyError(
            f"The label '{label_name}' is not a column in the DataFrame."
        ) from e


def min_max_scale_and_write(
    X_train: pd.DataFrame, X_test: pd.DataFrame, write_path: str = "output"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales the training and testing data using Min-Max scaling and writes the normalized data.

    Parameters:
    - X_train (pd.DataFrame): Training data to scale.
    - X_test (pd.DataFrame): Testing data to scale.
    - features (List[str]): List of feature names to be used for the columns of the normalized DataFrames.
    - write_path (str, optional): The directory path where the scaled datasets will be written.
    """
    min_max_scaler = MinMaxScaler().fit(X_train)
    X_train_normalized = min_max_scaler.transform(X_train)
    X_test_normalized = min_max_scaler.transform(X_test)

    features = list(X_train.columns)

    pd.DataFrame(X_train_normalized, columns=features).to_csv(
        f"{write_path}/train_data_normalized.csv", index=False
    )
    pd.DataFrame(X_test_normalized, columns=features).to_csv(
        f"{write_path}/test_data_normalized.csv", index=False
    )

    return X_train_normalized, X_test_normalized
