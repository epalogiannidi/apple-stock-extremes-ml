import logging
from datetime import datetime

from apple_stock_extremes_ml import ROOT_DIR as basename

__MAJOR__ = 1
__MINOR__ = 0
__PATCH__ = 0

__title__ = basename.replace("-", "_")
__version__ = ".".join([str(__MAJOR__), str(__MINOR__), str(__PATCH__)])
__summary__ = (
    "Predicting Extreme Events in Apple Stock Prices using Random Forest and TCNN"
)
__author__ = "Elisavet Palogiannidi"
__copyright__ = f"Copyright (C) {datetime.now().date().year} {__author__}"
__email__ = "epalogiannidi@gmail.com"

if __name__ == "__main__":
    logging.info(f"{__version__=}")
