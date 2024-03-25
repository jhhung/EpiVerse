import HiConformer
#from HiConformer.utils.DataCrawl import Crawl_Avocado_assays,md5sumcheck
from HiConformer.utils.Logger import init_logger
from HiConformer.utils import files 
from HiConformer.utils.Callbacks_13states import PlotLearning,WarmUpCosineDecayScheduler,WeightAdjuster, RegularRender
from HiConformer.model.DataGenerator_13states import DictHiConformerDataGenerator
from HiConformer.model.HiConformer_share_13states import HiConformer
from HiConformer.utils.loss_13states import Shrinkage_loss,L1_L2_Corr_Loss,multilabel_focal_loss,weighted_pearson_r,weighted_cosine_similarity, F1_Score, weighted_distance_pearson_r
import tensorflow as tf
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm
import os
from shutil import copyfile
import yaml
import argparse

# Initialize the ArgumentParser
parser = argparse.ArgumentParser(description='HiConformer training configuration')
parser.add_argument('--config', type=str, default="../HiConformer/Experiments/Training_config.yaml", required=True, help="Path to the training configuration file.")
parser.add_argument('--device', type=str, default=0, help="CUDA visible devices. Default is 0.", required=True)
parser.add_argument('--version', type=str, default="V001", help="User defined version name. Default is V001.", required=True)
parser.add_argument('--tissue', type=str, default="IMR90_MboI", help="Training Tissue name.", required=True)
parser.add_argument('--outdir', type=str, default="../training_results", help="Output directory for training results. Default is ../training_results.", required=True)

# Parse the arguments
args = parser.parse_args()

os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
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

with open(args.config, 'r') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
seq1_config        = config["seq1_config"]
seq2_config        = config["seq2_config"]
avo1_config        = config["avo1_config"]
avo2_config        = config["avo2_config"]
window_config      = config["window_config"]
transformer_config = config["transformer_config"]
regressor_config   = config["regressor_config"]
chromHMM_config    = config["chromHMM_config"]
training_config    = config["training_config"]
training_config["Version"]    = args.version
training_config["TrainTissue"] = [args.tissue]
current_time = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
#Result_dir = f"/work/u9485344/jasper/pretrains/{training_config['Version']}_{current_time}"
Result_dir = f"{args.outdir}/{training_config['Version']}_{current_time}"
files.create_output_dirs(Result_dir)

logger = init_logger(Result_dir)
logger.info("Reading Training configurations.")
copyfile(f"{args.config}",f"{Result_dir}/Training_config.yaml")

TrainDatagenConfig = training_config.copy()
ValidDatagenConfig = training_config.copy()
EvalDatagenConfig  = training_config.copy()

TrainDatagenConfig['ChromList']  = TrainDatagenConfig['TrainChromList']
TrainDatagenConfig['Tissues']    = TrainDatagenConfig['TrainTissue']
TrainDatagenConfig['Samples']    = TrainDatagenConfig['TrainSample']
ValidDatagenConfig['ChromList']  = ValidDatagenConfig['ValidChromList']
ValidDatagenConfig['Tissues']    = ValidDatagenConfig['ValidTissue']
ValidDatagenConfig['Samples']    = ValidDatagenConfig['ValidSample']
EvalDatagenConfig['ChromList']   = [17,17]
EvalDatagenConfig['Tissues']     = ["IMR90_MboI"]
EvalDatagenConfig['Samples']     = ["J011"]
EvalDatagenConfig["Shuffle"]     = False

model = HiConformer(seq1_config        ,
                    seq2_config        ,
                    avo1_config        ,
                    avo2_config        ,
                    window_config      ,
                    transformer_config ,
                    regressor_config   ,
                    chromHMM_config    ,
                    training_config    )
model.summary()

def sample_weight(DistPercentile:bool=False):
	if DistPercentile:
		Percentiles = Percentile_Norm(model.input[5]).values.reshape(-1,1)
		return Percentiles
	return np.ones((training_config['BatchSize'],1))

if training_config['Loss'] == "shrinkage":
	train_loss = Shrinkage_loss(a=a,c=c)
elif training_config['Loss'] == "mse":
	train_loss = tf.keras.losses.MeanSquaredError()
elif training_config['Loss'] == "Huber":
	train_loss = tf.keras.losses.Huber(delta=training_config['HuberDelta'])
elif training_config['Loss'] == "mae":
	train_loss = tf.keras.losses.MeanAbsoluteError()
elif training_config['Loss'] == "L1_L2_Loss":
	train_loss = L1_L2_Corr_Loss(W_l1=0.1)
elif training_config['Loss'] == "L1_L2_Distance_Corr_Loss":
    train_loss = L1_L2_Corr_Loss(W_l1=0.1)
else:
	raise RuntimeError("Unspecified loss for interaction frequency.")

if training_config["PretrainedWeight"]:
    logger.info("Load Pretrained checkpoint from " + training_config["PretrainedWeight"])
    model.load_weights(training_config["PretrainedWeight"])
else:
    logger.info("Train model from scratch!")

# Dataloader
train_DataGen = DictHiConformerDataGenerator(TrainDatagenConfig)
valid_DataGen = DictHiConformerDataGenerator(ValidDatagenConfig)
EvalDatagenConfig['PeakOversample'] = True
eval_DataGen  = DictHiConformerDataGenerator(EvalDatagenConfig, eval = True)

for data in train_DataGen:
	break
for data in valid_DataGen:
	break

Interaction_weight      = K.variable(1-training_config["ChromHMMWeight"])
bin1_ChromHMM_weight    = K.variable(training_config["ChromHMMWeight"])
bin2_ChromHMM_weight    = K.variable(training_config["ChromHMMWeight"])
loss_weights = {"Interaction_Frequency":Interaction_weight,
				"Bin1_ChromHMM_State":bin1_ChromHMM_weight,
				"Bin2_ChromHMM_State":bin2_ChromHMM_weight}

sample_count = 1e7 # num of batches
total_steps  = int(training_config['Epochs'] * sample_count)
warmup_steps = int(training_config['WarmupEpoch'] * sample_count)

if training_config['StepsPerEpoch'] is None:
	training_config['StepsPerEpoch'] = len(train_DataGen)

warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base   = training_config['LearningRateBase'],
										total_steps          = training_config['Epochs'] * training_config['StepsPerEpoch'],
										warmup_learning_rate = training_config['WarmupLearningRate'],
										warmup_steps         = training_config['WarmupEpoch'] * training_config['StepsPerEpoch'],
										hold_base_rate_steps = 0)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=Result_dir, histogram_freq=50)
callback_list=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.01, 
                                                patience=training_config['EarlyStopEpoch'],
                                                verbose=2,
                                                mode='auto',
                                                baseline=None, 
                                                restore_best_weights=False),
               warm_up_lr,
               tensorboard_callback,
               tf.keras.callbacks.ModelCheckpoint(filepath=Result_dir+"/checkpoints/ckpt-{epoch:05d}.h5",
                                                  period=training_config['SaveFreq'],
                                                  save_weights_only=True),
               RegularRender(eval_DataGen, Result_dir, freq = 50)]

def train(training_generator,validation_generator,callback_list,model,epochs):
    history = model.fit(    x=training_generator,
                            validation_data = validation_generator,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callback_list,
                            shuffle=False,
                            class_weight=None,
                            sample_weight=None,
                            initial_epoch=0,
                            steps_per_epoch  = training_config["StepsPerEpoch"],
                            validation_steps = training_config["StepsPerEpoch"],
                            max_queue_size=16,
                            workers=32,
                            use_multiprocessing=False
                          )    
    return model

# New Class Weight
class_weight = np.array([15.29372745,  5.70569221,  1.8277975 ,  2.06775414,  3.92258083,
                         4.85224877,  2.02363587,  3.75875014, 33.83250442,  6.63336985,
                         11.36640496, 17.70516168,  7.66416041])

logger.info(f"Weight for ChromHMM states:{class_weight}")

model.compile(optimizer='adam', 
              loss={'Interaction_Frequency':train_loss,
                    'Bin1_ChromHMM_State':multilabel_focal_loss(class_weight, epsilon = 1.e-7, gamma = 3.5),
                    'Bin2_ChromHMM_State':multilabel_focal_loss(class_weight, epsilon = 1.e-7, gamma = 3.5),
                    },
               metrics={'Interaction_Frequency':[weighted_pearson_r, weighted_distance_pearson_r],
                        'Bin1_ChromHMM_State': [F1_Score()],
                        'Bin2_ChromHMM_State': [F1_Score()]},
               loss_weights={'Interaction_Frequency': 1,
                             'Bin1_ChromHMM_State': training_config["ChromHMMWeight"],
                             'Bin2_ChromHMM_State': training_config["ChromHMMWeight"]})

trained_model = train(train_DataGen,
                      valid_DataGen,
                      callback_list,
                      model,
                      training_config['Epochs'])
#Save structure
json_string = trained_model.to_json()
with open(f"{Result_dir}/Model_structure.json", "w") as text_file:    
    text_file.write(json_string)
    text_file.close()
#Save weights
trained_model.save_weights(f"{Result_dir}/weights.h5")

