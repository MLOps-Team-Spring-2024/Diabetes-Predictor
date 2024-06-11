
# Spring 2024 MLOps Team Project: Diabetes-Predictor

## Team Members
Herny deBuchananne  
Matthew Soria  
Allan Guan  

## Project Description
Diabetes is a significant health concern in the United States, affecting millions of individuals. According to the latest statistics, about 37.3 million Americans, or 11.3% of the population, are living with diabetes. This condition, which can lead to severe health complications such as heart disease, kidney failure, and blindness, is also a major cause of disability and mortality. The cost of managing diabetes is substantial, with billions spent each year on healthcare services, medications, and lost productivity. Efforts to improve diabetes management and prevention are crucial in addressing this public health issue .

For our project we decided to create a classifier to identify diabetes in patients. Our dataset includes Age, Sex, HighChol, CholCheck, BMI, Smoker, HeartDiseasorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, DiffWalk, Stroke, HighBP and Diabetes. Based on this dataset we will classify patients as non-diabetic or diabetic. In order to classify patients we will use the Gradient Boosting machine learning technique. The framework we chose for Gradient Boosting was XGBoost.

The model primary objective is to facilitate the development and validation of predictive models that can accurately identify individuals at high risk of developing diabetes. By leveraging the comprehensive data collected from the Behavioral Risk Factor Surveillance System (BRFSS) of 2015, which includes a wide range of health indicators and demographic variables. Our team will apply machine learning techniques to improve early diagnosis and preventive care strategies. Our project aims to contribute to the broader field of healthcare analytics by providing a robust resource for studying the correlations and patterns that precede these serious health conditions.

### Project Scope and Objective
Additional scope and objective information can be found [here](./docs/source/overview.md)

## Phase 1
Phase 1 documentation can be found [here](./docs/source/phase_1.md)  

## Phase 2
Phase 2 documentation can be found [here](./docs/source/phase_2.md)  

## Phase 3
Phase 3 documentation can be found [here](./docs/source/phase_3.md)  

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
│     └── workflows            <- SubDirectory for the specific actions
│
├──logs                        <- Directory for logging and profiling logs
│  │
│  ├──logs                     <- SubDirectory for python.logging output
│  │
│  └── profiling               <- SubDirectory for all profiling logs
│       └── model_run          <- SubDirectory for tensorboard profiling logs
│
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
│   │   ├── logging
│   │   │   │
│   │   │   ├── logging.config <- Logging configurations
│   │   │   │
│   │   │   └── logs           <- Contain .log files
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

## Setting Up Environment 
#### Using Poetry

The virtual environment for this project uses Poetry, which needs to be set up on your local machine.

```console
pip install poetry
```

When Poetry is set up, a `pyproject.toml` file and a `poetry.lock` file are created if it is the first time, however there is already a `.toml` file created inour case.
To create the virtual environment and corresponding packages for the project, you must install the dependencies and environment itself. We also want to keep our virtual environment
within the project itself. To do this, run the following two commands

```console
poetry config virtualenvs.in-project true
poetry install
```

This will create your `.venv` file, which is our virtual environment.

To run the code within the environment, run

```console
poetry shell
```

Now you can run the project within this environment.

If you want to add dependencies to the project, it is as simple as running

```console
poetry add <dependency>
```

Which will automatically add the dependency to the `pyproject.toml` file. 
Removing it is just as simple

```console
poetry remove <dependency>
```

To exit the virtual environment you can run 

```console
exit
```

This doesn't necessarily deactive the environment. To do this you must the following command within the shell.

```console
deactivate
```

## Dependencies
- Make
- Python 3.11
- Poetry

The python package dependencies are automatically managed by Poetry
app:
* pandas = "^2.2.2"
* scikit-learn = "^1.4.2"
* xgboost = "^2.0.3"
* hydra-core = "^1.3.2"
* rich = "^13.3.2"
* wandb = "^0.17.0"
* torch = "^2.3.0"
* snakeviz = "^2.2.0"
* matplotlib = "^3.9.0"
* pydantic = "^2.7.3"
* google-cloud-storage = "^2.16.0"

dev:
* pytest = "^7.0"
* jupyterlab = "^4.1.8"
* ruff = "^0.4.2"
* mypy = "^1.10.0"
* dvc = "^3.51.2"
* dvc-gs = "^3.0.1"
* isort = "^5.13.2"
* interrogate = "^1.7.0"
* pre-commit = "^3.7.1"

## Contributions
Doc file containing detailed task and contribution  
[Part 1](/docs/project_1_tasks.txt)  
[Part 2](/docs/project_2_tasks.txt)  
[Part 3](/docs/project_3_tasks.txt)
