import os

_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")
_HYDRA_CONFIG = os.path.join(_PROJECT_ROOT, "mlops_team_project", "models", "config", "experiment", "baseline.yaml")
