import logging
import logging.config
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import omegaconf
import pandas as pd
import wandb
import xgboost as xgb
from hydra import compose, initialize
from omegaconf import OmegaConf
from rich.logging import RichHandler
from omegaconf.dictconfig import DictConfig
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from mlops_team_project.src.preprocess import (
    min_max_scale_and_write,
    train_test_split_and_write,
)


@dataclass
class ModelResponse:
    train_accuracy: float
    test_accuracy: float


def main(config: DictConfig, track_wandb: bool, wandb_project_name: str) -> None:
    """
    Main function that runs the necessary steps for modeling

    Args:
        config: hydra config which includes hyper parameters for xgboost
        track_wandb: boolean to determine if Weights and Biases is used
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

    model_response = model(
        X_train=X_train_normalized,
        X_test=X_test_normalized,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=hydra_params,
    )

    if track_wandb:
        wandb.init(project=wandb_project_name)
        wandb_config = wandb.config
        wandb_config.config = hydra_params
        wandb.log({"Train accuracy": model_response.train_accuracy})
        wandb.log({"Test accuracy": model_response.test_accuracy})


def model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    hyperparameters: omegaconf.dictconfig.DictConfig,
    target_names: List[str] = ["non-diabetic", "diabetic"],
) -> ModelResponse:
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

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Training: {train_accuracy}, Testing: {test_accuracy}\n")
    print(classification_report(y_test, base_model_preds, target_names=target_names))

    logging.info(f"cv scores = {cv_scores}\ncv scores avg = {cv_scores.mean()}\nTraining: {model.score(X_train, y_train)}, Testing: {model.score(X_test, y_test)}")
    
    logging.info(classification_report(y_test, base_model_preds, target_names=target_names))
    
    return ModelResponse(train_accuracy, test_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for xgboost model.")

    parser.add_argument(
        "--hydra_experiment",
        type=str,
        default="baseline",
        help="Hydra experiment yaml file",
    )
    parser.add_argument(
        "--wandb", type=bool, default=False, help="Track model with Weights and Biases"
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="se489-project",
        help="Project name for Weights and Biases",
    )

    args = parser.parse_args()

    print(f"hydra experiment = {args.hydra_experiment}")

    with initialize(version_base=None, config_path="config"):
        hydra_params = compose(overrides=[f"+experiment={args.hydra_experiment}"])

        main(hydra_params, args.wandb, args.wandb_project_name)
