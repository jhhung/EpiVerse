import logging
import os
import sys


def init_logger(output_dir: str) -> logging.Logger:
    """
    [summary]

    Args:
        output_dir (str):  Path to save log files

    Returns:
        logging.Logger: 
    """
    log_format = "[%(levelname)s] - %(asctime)s - %(funcName)s - %(message)s"
    log_dateformat = "%Y-%m-%d %H:%M:%S"

    Logger_path = os.path.join(output_dir, "training.log")

    logging.basicConfig(format=log_format, datefmt=log_dateformat, stream=sys.stdout, level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(Logger_path,mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # create a logging format
    formatter = logging.Formatter("[%(levelname)s] - %(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    
    # Logger Initialization
    logger.debug("HiConformer Logger is initialized.")
    logger.debug(f"Logger file is saved to {Logger_path}")

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)
