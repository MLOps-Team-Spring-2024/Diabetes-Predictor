# Spring 2024 MLOps Team Project: Diabetes-Predictor

## General Information

### Using Poetry
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

### Template Info
#### Project structure 
<details>
<summary>  
  see more
</summary>

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

Created using [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
</details>


<!-- see cookiecutter template for basic info on running the previous version of this -->


