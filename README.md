
# Spring 2024 MLOps Team Project: Diabetes-Predictor

## Team Members
Herny deBuchananne  
Matthew Soria  
Allan Guan  

## Project Description
Diabetes is a significant health concern in the United States, affecting millions of individuals. According to the latest statistics, about 37.3 million Americans, or 11.3% of the population, are living with diabetes. This condition, which can lead to severe health complications such as heart disease, kidney failure, and blindness, is also a major cause of disability and mortality. The cost of managing diabetes is substantial, with billions spent each year on healthcare services, medications, and lost productivity. Efforts to improve diabetes management and prevention are crucial in addressing this public health issue .

For our project we decided to create a classifier to identify diabetes in patients. Our dataset includes Age, Sex, HighChol, CholCheck, BMI, Smoker, HeartDiseasorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, DiffWalk, Stroke, HighBP and Diabetes. Based on this dataset we will classify patients as non-diabetic or diabetic. In order to classify patients we will use the Gradient Boosting machine learning technique. The framework we chose for Gradient Boosting was XGBoost.

The project primary objective is to facilitate the development and validation of predictive models that can accurately identify individuals at high risk of developing diabetes. By leveraging the comprehensive data collected from the Behavioral Risk Factor Surveillance System (BRFSS) of 2015, which includes a wide range of health indicators and demographic variables. Our team will apply machine learning techniques to improve early diagnosis and preventive care strategies. Our project aims to contribute to the broader field of healthcare analytics by providing a robust resource for studying the correlations and patterns that precede these serious health conditions.

### Project Scope and Objective
Additional scope and objective information can be found [here](./docs/source/overview.md)

## Data Overview
The dataset titled "Diabetes, Hypertension and Stroke Prediction" on Kaggle, created by Prosper Chuks, is based on survey data from the Behavioral Risk Factor Surveillance System (BRFSS) for the year 2015. It includes 70,692 responses that have been cleaned for analysis. The data is structured to facilitate the prediction of diabetes, hypertension, and stroke using various health indicators. This dataset is particularly valuable for developing machine learning models aimed at predicting these conditions.

For more detailed information, you can view the dataset directly on Kaggle: https://www.kaggle.com/datasets/prosperchuks/health-dataset


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

## Data Preprocessing
We have python modules to prepare the data. The code contains two functions. The first function splits the data into training and testing sets. The second function normalizes the data. The data was very clean from Kaggle.

>[mlops_team_project/src/preprocess.py](mlops_team_project/src/preprocess.py)

These functions are called from the main entryp point of the model 

>[mlops_team_project/models/xgboost_model.py](mlops_team_project/models/xgboost_model.py)

So when the model is invoked the steps will automatically run.

## EDA
We first started working with our model in Jupyter notebooks. You can start the notebook environment through make with

```
make run_jupyter
```

We didn't find any missing values or any non continuous numbers in our dataset to start. The dataset was pretty clean that we got from kaggle.

![eda1](images/eda1.png) 

However we did need to normalize the data so we used the MinMax scaler from scikit learn. The dataset also needed to be split into training and testing - and we used the scikit learn train_test_split function for that. Once that was finished we started our baseline model and experiment models.

>[notebooks/1_modeling.ipynb](notebooks/1_modeling.ipynb)

We set up our modeling function so it could pickup hydra config when it runs in production through decorators and it also could be injected in notebooks with different experiments from hydra. This allows use maximum flexibility in the notebooks to iterate quickly.  

```python
# run xgboost with exp1 params
with initialize(version_base=None, config_path="../mlops_team_project/models/config"):
    hydra_params = compose(overrides=["+experiment=exp1"])
    print(hydra_params)

    model(
        X_train=X_train_normalized,
        X_test=X_test_normalized,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=hydra_params.experiment,
    )
```

## Model
For our model we chose to use Gradient Boosting since we are predicting outcomes. Gradient Boosting is known to perform well on prediction tasks and allow for flexibitiy via different parameters.

Gradient boosting uses an ensemble technique - and typically it's creating multipe learners in the form of decision trees.  

As it iterates through the trees it's attempting to minimize errors and improve the accuracy.  

We used XGBoost as our framework to run this type of model.

## Steps to Replicate Training and Evaluation
When you run

```
make run_model
```

or

```
poetry run python mlops_team_project/models/xgboost_model.py
```

the model with split the dataset into train/testing sets and normalize. So preprocessing is automated through the entry point. We will optimize this in later tasks to only happen once when we start automatically chaining different tasks together.  

We are using Hydra for our hyperparameter tuning. We have a baseline configuration that runs without tuning any parameters and we have an experiment that tunes paramaters.  

>[mlops_team_project/models/config/default.yaml](mlops_team_project/models/config/default.yaml)  

When you want to run a new experiment you either change the value in the file above. If the experiment doesn't exist it first needs to be added to the experiments folder in `mlops_team_project/models/config/experiment`

We used Cross validation for our evaluation process with k folds = 5.  

We mainly our looking at accuracy scores to choose the best - model while making sure the model is not overfitting.  

Our baseline model outperformed out tuned model. Baseline had a 78%/74% split between train and test. This shows it didn't overfit and generalized well. However our tuned model had a split of 83%/73% so it was showing more overfitting and the testing accuracy went down slightly. So we would move forward with out basline model based on this evaluation.  

Evaluation results below.

Baseline Model  

![baseline](images/baseline.png)  

Tuned Model 
![exp1](images/exp1.png)  

## GitHub Actions

To enhance collaboration and cohesiveness of our coding standards in our distributed environment, three GitHub Actions were created. The most basic action simply automatically creates a Pull Request review email upon the opening of a new Pull Request, automatically sending the request to other teammates when it has been pushed, which ensures that the PRs do not miss being reviewed.

To establish cohesive coding standards and documentation standards, the two other GitHub actions implement Ruff, for styling, and MyPy for automated testing. The action for Ruff automatically checks all our code at PR opening and at every push, and if there is a need for reformatting, the action will automatically run ` fix ` according to our defined standards, and commit the fix accordingly. The MyPy action runs against all python code, scanning for any issues with formatting, typing, etc. and if any are detected, the PR will be unable to merge until these issues are remediated.

## Project structure 
<details>

The directory structure of the project looks like this:

```txt

├── Makefile                   <- Makefile with convenience commands like `make data` or `make train`
├── README.md                  <- The top-level README for developers using this project.
├── data
│   ├── processed              <- The final, canonical data sets for modeling.
│   └── raw                    <- The original, immutable data dump.
│
├── docs                       <- Documentation folder
│   │
│   ├── index.md               <- Homepage for your documentation
│   │
│   ├── mkdocs.yml             <- Configuration file for mkdocs
│   │
│   └── source/                <- Source directory for documentation files
│
├── .github                    <- Source directory for GitHub Actions and configurations
|     └── workflows            <- SubDirectory for the specific actions
|
├── models                     <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks                  <- Jupyter notebooks.
│
├── pyproject.toml             <- Project configuration file
│
├── poetry.lock                <- Poetry lock file that contains locked dependency versions
│
├── tests                      <- Test files
│
├── mlops_team_project         <- Source code for use in this project.
│   │
│   ├── models                 <- Model implementations, training script and prediction script
│   │   │
│   │   ├── config             <- Folder container hydra config files
│   │   │
│   │   ├── xgboost_model.py   <- Entry point that runs our xgboost model
│   │
│   ├── src                    <- Scripts to create exploratory and results oriented visualizations
│       |
│       └── preprocess.py      <- Functions that split into train/test and normalize
│   
└── LICENSE                    <- Open-source license if one is chosen
```

</details>

## What to improve on in future iterations
Right now when the model runs we are splitting up the full dataset into train/test and normalizing every time. We only need to do this once. However since we don't know how we are going to output our model and since these tasks run very quickly we decided to wait until we have more information in later phases to optimize this part of our code.

## Dependencies
- Make
- Python 3.11
- Poetry

The python package dependencies are automatically managed by Poetry
* jupyterlab 4.1.8
* pandas 2.2.2
* scikit-learn 1.4.2
* xgboost 2.0.3
* hydra-core 1.3.2
* ruff 0.4.2
* mypy 1.10.0

## Architectural Overview
![Overview](images/Overview.jpg)  

### Contributions
Doc file containing detailed task and contribution
[Part 1](/docs/project_1_tasks.txt)


