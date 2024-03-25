# %%
import os
#import cv2
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib as mpl
import numpy as np
from datetime import datetime
from scipy.stats import zscore
from tensorflow.keras import backend as K
from MIRNet import mirnet_model, read_image, get_dataset
import argparse
# %%
def set_gpu_status(device):
    os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/local/cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["CUDA_HOME"]="/usr/local/cuda"
    os.environ['NUMEXPR_NUM_THREADS']="32"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=10)

def pearson_correlation(y_true, y_pred):
    # Mean of the true and predicted values
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    
    # Numerator and denominator for Pearson correlation
    numerator = tf.reduce_sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_true_mean)) * tf.reduce_sum(tf.square(y_pred - y_pred_mean)))
    
    # Avoid division by zero
    correlation_coefficient = tf.cond(tf.equal(denominator, 0), lambda: 0.0, lambda: numerator / denominator)
    
    return correlation_coefficient

def diagonal_correlation_loss(y_true, y_pred):
    """
    Calculates the correlation between the diagonals of the true and predicted matrices., 
    from the 4th diagonal to 198 diagonals the batch image size is (B, 200, 200, 1)
    """
    correlations = []
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    
    for i in range(4, 200): # 4th diagonal to 199th diagonal
        diag_true = tf.linalg.diag_part(y_true, k=i)
        diag_pred = tf.linalg.diag_part(y_pred, k=i)
        
        # Calculating correlation for the diagonals
        mean_true = tf.reduce_mean(diag_true)
        mean_pred = tf.reduce_mean(diag_pred)
        
        cov = tf.reduce_mean((diag_true - mean_true) * (diag_pred - mean_pred))
        std_true = tf.math.reduce_std(diag_true)
        std_pred = tf.math.reduce_std(diag_pred)
        
        corr = cov / (std_true * std_pred + 1e-8) # added small value to avoid division by zero
        """
        if i == 199:
            tf.print(corr)
            tf.print(cov)
            tf.print(diag_true.shape)
        """
        correlations.append(corr)
    
    mean_corr = tf.reduce_mean(correlations)
    
    return 1 - mean_corr

def surrogate_loss(weight_charb=1.0, weight_diag_corr=1.0):
    def loss(y_true, y_pred):
        charb_loss = charbonnier_loss(y_true, y_pred)
        diag_corr_loss = diagonal_correlation_loss(y_true, y_pred)
        #print(diag_corr_loss)
        #print(charb_loss)
        return charb_loss + diag_corr_loss
    # Combine the two losses. You can also use a weighting factor if needed.
    return loss
  
def init_parser():
    parser = argparse.ArgumentParser(description='MIRNet for Hi-C data enhancement.')
    parser.add_argument('--image_size', type=int, default=200, help='Size of the input images.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--train_data_dir', type=str, required=True, help='Directory containing training data.')
    parser.add_argument('--val_data_dir', type=str, required=True, help='Directory containing validation data.')
    parser.add_argument('--result_dir', type=str, default='../training_results', help='Directory to save the results and model checkpoints.')
    parser.add_argument("--device", type=str, default="0", help="GPU device to use.")
    return parser.parse_args()
  
if __name__ == "__main__":
    random.seed(10)
    args = init_parser()
    set_gpu_status(args.device)
    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    TRAIN_DATA_DIR = args.train_data_dir
    VAL_DATA_DIR = args.val_data_dir
    current_time = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    RESULT_DIR = f"{args.result_dir}/MIRNet_{current_time}"
    os.makedirs(RESULT_DIR)

    ## training dataset
    train_low_light_images = sorted(glob(f"{args.train_data_dir}/imputation/*.npy"))
    train_enhanced_images = sorted(glob(f"{args.train_data_dir}/ground_truth/*.npy"))
    ## validation dataset
    val_low_light_images = sorted(glob(f"{args.val_data_dir}/imputation/*.npy"))
    val_enhanced_images = sorted(glob(f"{args.val_data_dir}/ground_truth/*.npy"))
    ## get dataset
    train_dataset = get_dataset(train_low_light_images, train_enhanced_images)
    val_dataset = get_dataset(val_low_light_images, val_enhanced_images)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    model = mirnet_model(num_rrg=2, num_mrb=2, channels=64)
    print(model.summary())
    # %%
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer, loss=surrogate_loss(1.0, 1.0),  metrics=[peak_signal_noise_ratio, pearson_correlation, diagonal_correlation_loss]
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=RESULT_DIR, histogram_freq=1)
    early_stopper = keras.callbacks.EarlyStopping(
        monitor='val_diagonal_correlation_loss', 
        patience=50, 
        verbose=1, 
        mode='max', 
        restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_diagonal_correlation_loss",
                factor=0.9,
                patience=20,
                verbose=1,
                min_delta=1e-7,
                mode="max"),
            tf.keras.callbacks.ModelCheckpoint(filepath=RESULT_DIR+ "-{epoch:03d}.h5",
                save_freq="epoch",
                save_best_only=True,  # Save only when the metric improves
                save_weights_only=True),
            tensorboard_callback,
            early_stopper
        ],
        epochs=EPOCHS,
        shuffle=True,
    )