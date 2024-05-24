import logging
import logging.config
from pathlib import Path
from typing import List

import hydra
import numpy as np
import omegaconf
import pandas as pd
import xgboost as xgb
from omegaconf import OmegaConf
from rich.logging import RichHandler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from mlops_team_project.src.preprocess import (
    min_max_scale_and_write,
    train_test_split_and_write,
)


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(config) -> None:
    """
    Main function that runs the necessary steps for modeling

    Args:
        config: hydra config which includes hyper parameters for xgboost
    """
    logging.config.fileConfig(Path(__file__).resolve().parent / "logging" / "logging.config")
    logger = logging.getLogger(__name__)
    logger.root.handlers[0] = RichHandler(markup=True)
    
    print(f"conf = {OmegaConf.to_yaml(config)}")
    hydra_params = config.experiment

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
        hyperparameters=hydra_params,
    )


def model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    hyperparameters: omegaconf.dictconfig.DictConfig,
    target_names: List[str] = ["non-diabetic", "diabetic"],
) -> None:
    """
    Runs the XGBoost model.

    Args:
        X_train: train dataset.
        X_test: test dataset.
        y_train: labels for training.
        y_test: labels for test.
    """
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        random_state=hyperparameters.seed,
        n_estimators=hyperparameters.n_estimators,
    )

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    model.fit(X_train, y_train)
    base_model_preds = model.predict(X_test)

    print(f"cv scores = {cv_scores}")
    print(f"cv scores avg = {cv_scores.mean()}")
    print(
        f"Training: {model.score(X_train, y_train)}, Testing: {model.score(X_test, y_test)}\n"
    )
    print(classification_report(y_test, base_model_preds, target_names=target_names))

    
    
    logging.info(f"cv scores = {cv_scores}\ncv scores avg = {cv_scores.mean()}\nTraining: {model.score(X_train, y_train)}, Testing: {model.score(X_test, y_test)}")
    
    logging.info(classification_report(y_test, base_model_preds, target_names=target_names))

if __name__ == "__main__":
    main()
