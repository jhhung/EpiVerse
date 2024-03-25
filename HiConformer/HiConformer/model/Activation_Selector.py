
import tensorflow as tf 
import tensorflow_addons as tfa
from   tensorflow.keras import backend as K
from   HiConformer.utils.Logger import get_logger

def get_activation_function():
    logger = get_logger()
    try:
        from tensorflow.nn import gelu as activation
    except ImportError as Error:
        logger.warning("GeLU is not supported. ReLU is applied")
        from tensorflow.nn import relu as activation
    return activation