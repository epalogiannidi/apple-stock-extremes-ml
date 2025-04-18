## Development 🧱

### Set up the environment 🐍
1. Create the environment: ``` pipenv --python 3.10 ```
2. Activate the environment: ``` pipenv shell ```
3. Install dependencies ``` pipenv install --dev```

### Development support 💁
Execute the following commands to assure code quality:

- Linter
```commandline
make lint
```

- Code formatter
```commandline
make black
```

- Sort imports
```commandline
make isort
```

- Mypy/ Typehints
```commandline
make mypy
```

- Pytests
```commandline
make tests
```

- Create docker image
```commandline
make docker-build
```

## Track experimental procedure 🕵️‍♀️
- Start mlflow server. After executing the UI is accessible in: http://127.0.0.1:5000/

``` commandline
bash start_mlflow_server.sh
```

## Execution 🚀

Before executing any of the following commands, open and edit configs/config.yaml.
Change the name and the (hyper)parameters to execute a different experiment.
api_predict variable contains the saved models that are going to be used for the predictions (model_predict/inference)

### Command Line & Docker 

For executing using docker, first load the docker image:
```commandline
docker load -i apple_stock_extremes_ml.tar
```

*Tasks not supported by docker*: predict-rf, predict-tcnn

- Data processing and analysis
```commandline
python -m apple_stock_extremes_ml --task data-analysis
```
```Docker:
docker run --rm apple_stock_extremes_ml --task data-analysis
```
- Train a random forest model

```commandline
python -m apple_stock_extremes_ml --task train-rf
```
```Docker
docker run apple_stock_extremes_ml --task train-rf
```

- Train tcnn model
```commandline
python -m apple_stock_extremes_ml --task train-tcnn
```
```Docker
docker run apple_stock_extremes_ml --task train-tcnn
```

- Hyperparameter tuning of the tcnn model
```commandline
python -m apple_stock_extremes_ml --task tune-tcnn
```

- Analyze and train a random forest and a tcnn model
```commandline
python -m apple_stock_extremes_ml --task end-to-end
```
```Docker
docker run apple_stock_extremes_ml --task end-to-end
```

- Get predictions using random forest model
```commandline
python -m apple_stock_extremes_ml --task predict-rf
```

- Get predictions using random tcnn model
```commandline
python -m apple_stock_extremes_ml --task predict-tcnn
```

### Docker 🐳
- Save a docker image
```commandline
 docker save -o apple_stock_extremes_ml.tar apple_stock_extremes_ml
```

-Load a docker image
```commandline
docker load -i apple_stock_extremes_ml.tar
```


### Implementation details 🔍
- Dependency management: pipenv
- A yaml configuration file is necessary to pass the inputs to the execution scripts
- A global logger has been set up to keep the logs of each execution. A file with the execution date is being created to save the logging information
- Experiments are being tracked using mlflow
- Any plot that is generated by the code is saved under plots directory
- pytest has been set up to test the functionality

### Project structure 🌳

- **apple_stock_extremes_ml**:
  - The package that contains the main source code
  - **__init__.py** is used to prepare the environment (set up logger, create directories, read configurations)
  - **__main__.py** is the executor wrapper that allows to run the script as a module for all the available tasks
  - **train_tcnn.py** is responsible for training the tcnn model
  - **data_handler.py** does the data analysis and preprocessing
  - **models** directory contains the model architectures:
    - **rf.py** for random forest
    - **tcnn.py** for tcnn
    - **tcnn_tuning.py** for tuning the model
  - The rest of the python files and directories within this directory contain supplementary information that is necessary for the correct and well structured functionality of the code
- **configs**:
  - **config.yaml** is the main configuration file. Modify this file to execture a different experiment
  - logger.conf and mypy.ini are the configuration files required for the logger and the mypy library
- **logs**:
  - Automatically generated directory for saving the log files in the format: DDMMYYY-HH:MM:SS.log. A log file is generated with each code execution.
- **plots**:
  - Automatically generated directory to save plots that are being created during the various steps of the execution
- **tests**:
  - Contains the code for testing the functionality using pytest
- **mlruns**:
  - Stores the experiments saved on MLFLOW
- Pipfile: contains the dependencies
- Pipfile.lock: automatically generated file for dependency handling
- Makefile: contains the development support commands
- CHANGELOG.md: the file for reporting the additions/changes etc
- CONTRIBUTING.md: a file to report contributing and collaborating instructions that are helpful during development
- Dockerfile: the file used to build a docker image
- [apple_stock_extremes_ml.tar](apple_stock_extremes_ml.tar) The docker image
- report.pdf: A report describing the project process

## Known issues/bugs 🐛
- pytest coverage is 50%
- currently, with each execution an additional experiment is being created. Look only at experiments within Random_Forest_Experiment and TCNN_Experiment
- Docker execution doesn't work for  predict-rf and predict-tcnn
