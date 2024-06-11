import argparse
import io
import logging
import logging.config
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import wandb
import xgboost as xgb
from google.cloud import storage
from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from rich.logging import RichHandler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from mlops_team_project.src.preprocess import (
    min_max_scale_and_write,
    train_test_split_and_write,
)


@dataclass
class ModelResponse:
    train_accuracy: float
    test_accuracy: float


BUCKET_NAME = "mlops489-project"

client = (
    storage.Client("mlops489-425700")
    if (os.environ.get("RUN_CML") != "true" and os.environ.get("NO_STORAGE") != "true")
    else None
)


def main(
    config: DictConfig,
    track_wandb: bool,
    wandb_project_name: str,
    profile_perf: bool,
    read_local_data: bool,
) -> None:
    """
    Main function that runs the necessary steps for modeling

    Args:
        config: hydra config which includes hyper parameters for xgboost
        track_wandb: boolean to determine if Weights and Biases is used
    """
    logging.config.fileConfig(
        Path(__file__).resolve().parent / "logging" / "logging.config"
    )
    logger = logging.getLogger(__name__)
    logger.root.handlers[0] = RichHandler(markup=True)

    logger.info(f"conf = {OmegaConf.to_yaml(config)}")
    hydra_params = config.experiment

    if read_local_data:
        df = pd.read_csv("data/raw/diabetes_data.csv")
    else:
        df = pd.read_csv(io.BytesIO(read_from_google("data/raw/diabetes_data.csv")))

    X_train, X_test, y_train, y_test = train_test_split_and_write(
        df=df, write_path="data/processed"
    )

    X_train_normalized, X_test_normalized = min_max_scale_and_write(
        X_train=X_train, X_test=X_test, write_path="data/processed"
    )
    """
        NOTE: to profile over multiple runs, make sure to include prof.step() on each iteration
        ex: when looping, on each iteration include prof.step()
    """

    if profile_perf:
        prof_log: str = "./logs/profiling/model_run"

        curr_env = os.getenv("IN_CONTAINER", False)

        if curr_env:
            prof_log = os.getenv("PERF_DIR", prof_log)

        # begin profile block
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(prof_log),
        ) as prof:
            model_response = model(
                X_train=X_train_normalized,
                X_test=X_test_normalized,
                y_train=y_train,
                y_test=y_test,
                hyperparameters=hydra_params,
            )

            if track_wandb:
                wandb_api_key = os.getenv("WANDB_API_KEY")
                if wandb_api_key:
                    wandb.login(key=wandb_api_key)
                wandb.init(project=wandb_project_name)
                wandb_config = wandb.config
                wandb_config.config = hydra_params
                wandb.log({"Train accuracy": model_response.train_accuracy})
                wandb.log({"Test accuracy": model_response.test_accuracy})
            prof.step()
    else:
        model_response = model(
            X_train=X_train_normalized,
            X_test=X_test_normalized,
            y_train=y_train,
            y_test=y_test,
            hyperparameters=hydra_params,
        )

        if track_wandb:
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key)
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

    print(X_train.shape)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    logging.info(
        f"cv scores = {cv_scores}\ncv scores avg = {cv_scores.mean()}\nTraining: {train_accuracy}, Testing: {test_accuracy}"
    )

    logging.info(classification_report(y_test, preds, target_names=target_names))

    if os.environ.get("RUN_CML") == "true":
        creat_cml_report(y_test, preds, target_names)
    else:
        save_model_to_google(model)

    return ModelResponse(train_accuracy, test_accuracy)


def read_from_google(file_name: str):
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)
    return blob.download_as_bytes()


"""
    Creates a CML report that will get posted as a comment on a PR
    similar to logging, but for added visibility during code review
"""


def creat_cml_report(y_test, preds, target_names: list[str]):
    report = classification_report(y_test, preds, target_names=target_names)
    with open("classification_report.txt", "w") as outfile:
        outfile.write(report)

    confmat = confusion_matrix(y_test, preds)
    display = ConfusionMatrixDisplay(confusion_matrix=confmat)
    fig, ax = plt.subplots(figsize=(10, 8))  # may want to update the size
    display.plot(ax=ax)

    plt.savefig("confusion_matrix.png")


def save_model_to_google(model):
    bytes_for_pickle = io.BytesIO()
    pickle.dump(model, bytes_for_pickle)
    bytes_for_pickle.seek(0)

    bucket = client.get_bucket(BUCKET_NAME)

    blob = bucket.blob("models/xgboost_model.pkl")
    blob.upload_from_file(bytes_for_pickle, content_type="application/octet-stream")


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
    parser.add_argument("--profile", type=bool, default=False)
    parser.add_argument("--read-local-data", type=bool, default=False)

    args = parser.parse_args()

    print(f"hydra experiment = {args.hydra_experiment}")

    with initialize(version_base=None, config_path="config"):
        hydra_params = compose(overrides=[f"+experiment={args.hydra_experiment}"])
        main(
            hydra_params,
            args.wandb,
            args.wandb_project_name,
            args.profile,
            args.read_local_data,
        )
