# import logging
# import sys
# from datetime import datetime

# def setup_logger():
#     logger = logging.getLogger('InferenceService')
#     logger.setLevel(logging.INFO)

#     # Create handlers
#     c_handler = logging.StreamHandler(sys.stdout)
#     f_handler = logging.FileHandler('/opt/ml/model/inference.log')
    
#     # Create formatters
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     c_handler.setFormatter(formatter)
#     f_handler.setFormatter(formatter)

#     # Add handlers to the logger
#     logger.addHandler(c_handler)
#     logger.addHandler(f_handler)
    
#     return logger

# logger = setup_logger()

import logging
import sys
from datetime import datetime
import os

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    
    # Ensure the directory exists for the log file
    os.makedirs('/opt/ml/model', exist_ok=True)
    f_handler = logging.FileHandler('/opt/ml/model/inference.log')
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger if they haven't been added already
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

# Function that serve script is looking for
def get_logger(name):
    return setup_logger(name)

# Maintain backwards compatibility for existing code
logger = setup_logger('InferenceService')