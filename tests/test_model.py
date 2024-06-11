from unittest.mock import Mock, patch

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from xgboost import XGBClassifier

from mlops_team_project.models.xgboost_model import model
from tests import _HYDRA_CONFIG


@pytest.fixture
def input_data():
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    target_names = ["non-diabetic", "diabetic"]
    return X_train, X_test, y_train, y_test, target_names

def test_model(input_data):
    """
        Unit test for model size
    """
    X_train, X_test, y_train, y_test, target_names = input_data

    with patch('mlops_team_project.models.xgboost_model.xgb.XGBClassifier') as xgb_model, \
            patch('mlops_team_project.models.xgboost_model.cross_val_score') as cv_score, \
            patch('mlops_team_project.models.xgboost_model.logging.info') as logging, \
            patch('mlops_team_project.models.xgboost_model.save_model_to_google') as gcloud, \
            patch('mlops_team_project.models.xgboost_model.classification_report') as classification_report:

        mock_model = Mock()
        mock_model.predict.return_value = np.random.randint(2, size=y_train.shape)
        mock_model.score.side_effect = [0.8, 0.7]
        xgb_model.return_value = mock_model
        cv_score.return_value = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
        cv_score.mean.return_value = 1.1
        hydra_config = OmegaConf.load(_HYDRA_CONFIG)

        response = model(X_train, X_test, y_train, y_test, hydra_config, target_names)

        assert response.train_accuracy == 0.8
        assert response.test_accuracy == 0.7
        assert mock_model.predict.return_value.shape == y_test.shape
