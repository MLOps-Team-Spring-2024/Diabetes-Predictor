
# Spring 2024 MLOps Team Project: Diabetes-Predictor

## Team Members
Herny deBuchananne
Matthew Soria
Allan Guan

## Data Overview
The dataset titled "Diabetes, Hypertension and Stroke Prediction" on Kaggle, created by Prosper Chuks, is based on survey data from the Behavioral Risk Factor Surveillance System (BRFSS) for the year 2015. It includes 70,692 responses that have been cleaned for analysis. The data is structured to facilitate the prediction of diabetes, hypertension, and stroke using various health indicators. This dataset is particularly valuable for developing machine learning models aimed at predicting these conditions.

For more detailed information, you can view the dataset directly on Kaggle: https://www.kaggle.com/datasets/prosperchuks/health-dataset

## Objective
The primary objective of the "Diabetes, Hypertension and Stroke Prediction" dataset is to facilitate the development and validation of predictive models that can accurately identify individuals at high risk of developing diabetes, hypertension, and stroke. By leveraging the comprehensive data collected from the Behavioral Risk Factor Surveillance System (BRFSS) of 2015, which includes a wide range of health indicators and demographic variables, researchers and data scientists can apply machine learning techniques to improve early diagnosis and preventive care strategies. This dataset aims to contribute to the broader field of healthcare analytics by providing a robust resource for studying the correlations and patterns that precede these serious health conditions.


## Setting Up Environment 
#### Using Poetry

The virtual environment for this project uses Poetry, which needs to be set up on your local machine.

`pip install poetry`

When Poetry is set up, a `pyproject.toml` file and a `poetry.lock` file are created if it is the first time, however there is already a `.toml` file created inour case.
To create the virtual environment and corresponding packages for the project, you must install the dependencies and environment itself. We also want to keep our virtual environment
within the project itself. To do this, run the following two commands

`poetry config virtualenvs.in-project true`

`poetry install`

This will create your `.venv` file, which is our virtual environment.

To run the code within the environment, run

`poetry shell`

Now you can run the project within this environment.

If you want to add dependencies to the project, it is as simple as running

`poetry add <dependency>`

Which will automatically add the dependency to the `pyproject.toml` file. 
Removing it is just as simple

`poetry remove <dependency>`

To exit the virtual environment you can run 

`exit`

This doesn't necessarily deactive the environment. To do this you must the following command within the shell.

`deactivate`

## Data 
We have python codes to prepare the data. The code contains two functions. The first function splits the data into training and testing sets. The second function normalizes the data. The data was very clean from Kaggle.

>[mlops_team_project/src/preprocess/preprocess.py](mlops_team_project/src/preprocess/preprocess.py)

## Architectural Overview
![Overview](https://github.com/MLOps-Team-Spring-2024/Diabetes-Predictor/tree/main/images/Overview.jpg)

## Project structure 
<details>

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mlops_test_cookiecutter  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```


</details>




## Steps to Replicate Training and Evaluation

TBA

## Dependencies

The dependencies are automatically managed by Poetry
* python 3.11
* jupyterlab 4.1.8
* pandas 2.2.2
* scikit-learn 1.4.2
* xgboost 2.0.3
* hydra-core 1.3.2

### Contributions
--link to doc file--


