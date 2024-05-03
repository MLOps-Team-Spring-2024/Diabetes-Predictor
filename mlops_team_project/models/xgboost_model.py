import hydra
import omegaconf
import pandas as pd
from omegaconf import OmegaConf

from mlops_team_project.src.model.model import model
from mlops_team_project.src.preprocess.preprocess import (
    min_max_scale_and_write,
    train_test_split_and_write,
)


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(config) -> None:
    """
    Main function that runs the necessary steps for modeling.

    Args:
        config: hydra config which includes hyper parameters for xgboost
    """
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


if __name__ == "__main__":
    main()
