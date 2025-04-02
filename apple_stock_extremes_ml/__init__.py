import logging.config
import os
import time

import yaml  # type: ignore

""" Define directories and files """
ROOT_DIR = os.getcwd()
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
APP_CONFIG = os.path.join(CONFIGS_DIR, "config.yaml")

""" Set up logger directory and format """
os.makedirs(LOGS_DIR, exist_ok=True)

log_name = time.strftime("%m%d%Y-%H:%M:%S")

logging.disable(logging.DEBUG)
logging.config.fileConfig(
    fname=os.path.join(CONFIGS_DIR, "logger.conf"),
    disable_existing_loggers=False,
    defaults={"logfilename": f"{LOGS_DIR}/{log_name}.log"},
)
logger = logging.getLogger(__name__)

os.makedirs(PLOTS_DIR, exist_ok=True)

""" Load configuration values"""
""" Load configuration values"""
try:
    with open(APP_CONFIG, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError
else:
    logger.info("Model info file loaded successfully.")

app_config = config
