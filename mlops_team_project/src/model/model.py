from typing import List

import numpy as np
import omegaconf
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


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
