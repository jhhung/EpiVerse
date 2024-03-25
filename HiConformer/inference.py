# %%
from HiConformer.utils.DataCrawl import Crawl_Avocado_assays,md5sumcheck
from HiConformer.utils.Logger import init_logger
from HiConformer.utils import files 
from HiConformer.utils.Callbacks_13states import PlotLearning,WarmUpCosineDecayScheduler,WeightAdjuster, RegularRender
from HiConformer.model.DataGenerator_13states_eval import DictHiConformerDataGenerator
from HiConformer.model.HiConformer_share_13states import HiConformer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import os
from shutil import copyfile
import tensorflow_addons as tfa
import yaml
import gc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import seaborn as sns
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
            print("[INFO] ", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)

def read_reference_seq(ref_seq_path):
    # %%
    from pyfaidx import Fasta
    ref = Fasta(ref_seq_path, 
                read_ahead            = 200000000)
    return ref
# %%
def create_model(model_config_path, eval_ckpt, tissue, sample):
    """
    read and create model config
    """
    print(f"[INFO] read model config from {model_config_path}")
    with open(model_config_path, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    seq1_config        = config["seq1_config"]
    seq2_config        = config["seq2_config"]
    avo1_config        = config["avo1_config"]
    avo2_config        = config["avo2_config"]
    window_config      = config["window_config"]
    transformer_config = config["transformer_config"]
    regressor_config   = config["regressor_config"]
    training_config    = config["training_config"]
    chromHMM_config    = config["chromHMM_config"]
    
    EvalDatagenConfig = training_config.copy()
    EvalDatagenConfig['ChromList']  = [21, 21]
    EvalDatagenConfig['Tissues']    = [tissue]
    EvalDatagenConfig['Samples']    = [sample]
    EvalDatagenConfig["Shuffle"]    = False
    print(f"[INFO] create model using model config")
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
    print(f"[INFO] load pretrained weight from {eval_ckpt}")
    model.load_weights(eval_ckpt)
    return model, EvalDatagenConfig

def convert_arr(sparse: list):
    arr = np.zeros(13)
    if len(sparse) != 2:
        sparse = [int(state) for state in sparse[1:-1].split(" ") if state != '']
        for state in sparse:
            arr[state] = 1
    return arr

def hiconformer_diagonal_builder(seq_len, chrom):
    """
    build a diagnoal according to 3DIV data format
    We choose valid diagnoal though following settings
    1. If bin1-bin2 Distance == 20000, find a vertical diagoanl line with 100 bins
    2. If bin1-bin2 Distance == 25000, also fina a vertical diagnoal line with 100 bins
    """
    diags = []
    for i in tqdm(range(seq_len)):
        for j in range(seq_len):
            diagnoal = []
            if (j - i == 4 and (i - 100) > 0 and (j + 100) < seq_len):
                start_bin1 = i
                start_bin2 = j
                for k in range(100):
                    diagnoal.append([start_bin1 - k, start_bin2 + k, 4 + 2*k])
            if (j - i == 5 and (i - 100) > 0 and (j + 100) < seq_len):
                start_bin1 = i
                start_bin2 = j
                for k in range(100):
                    diagnoal.append([start_bin1 - k, start_bin2 + k, 5 + 2*k])
            if (len(diagnoal) == 100):
                diags.append(diagnoal)
    df = pd.DataFrame(np.concatenate(diags, axis=0), columns=["bin1", "bin2", "Distance"])
    df = df * 5000
    df["rescaled_intensity"] = 0.0
    df["dist_foldchange"] = 0.0
    df["chr"] = f"chr{chrom}"
    return df

def init_parser():
    """
    HiConformer parser
    """
    parser = argparse.ArgumentParser(description='evaluate-plotter-insertion')
    parser.add_argument('--chrom', help='input permutation index', type=int)
    parser.add_argument('--tissue', help="input tissue name", type=str)
    parser.add_argument('--sample', help="input sample name", type=str)
    parser.add_argument("--REF_SEQ", help="input reference sequence path", type=str)
    parser.add_argument("--EVAL_CKPT", help="input evaluation checkpoint path", type=str)
    parser.add_argument("--MODEL_CONFIG", help="input model config path", type=str)
    parser.add_argument("--OUTPUT_DIR", help="input output directory", type=str)
    parser.add_argument("--device", help="input device", type=str)
    args = parser.parse_args()
    return args
# %%
if __name__ == "__main__":
    """
    #EVAL_CKPT      = "/mnt/nas_1/backup_20220825/DeepLearning/jasper/Results/A100_TrainIMR90_ValIMR90_IPDropConti_2320_2023_05_07_15_02_22/checkpoints/ckpt-03440.h5"
    #MODEL_CONFIG   = "/mnt/nas_1/backup_20220825/DeepLearning/jasper/Results/A100_TrainIMR90_ValIMR90_IPDropConti_2320_2023_05_07_15_02_22/Training_config.yaml"
    #REF_SEQ = "/mnt/nas_1/backup_20220825/DeepLearning/wenny/reference/hg38.fa"
     # evaluation tissues
    #EVAL_TISSUES=  ["GM12878_MboI", "K562", "kidney", "ovary", "pancreas", "prostate gland", "aorta", "psoas muscle"]
    #EVAL_SAMPLES =  ["J358", "J599", "J257", "J149", "J351", "J402", "J547", "J232"]
    """
    # pretrained model and model config file
    args = init_parser()
    set_gpu_status(args.device)
    # Load args
    EVAL_CKPT = args.EVAL_CKPT
    MODEL_CONFIG = args.MODEL_CONFIG
    REF_SEQ = args.REF_SEQ
    # ref seq hg38
    ref = read_reference_seq(REF_SEQ)
    # parse args
    chrom = args.chrom
    tissue = args.tissue
    sample = args.sample
    # start evaluation steps
    OUTPUT_DIR = f"{args.OUTPUT_DIR}/{tissue}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Load Model Weight and Config
    model, EvalDatagenConfig = create_model(MODEL_CONFIG, EVAL_CKPT, tissue, sample)
    EvalDatagenConfig['ChromList'] = [chrom, chrom]
    print(f"[INFO] Now impute {tissue}-chr{chrom}")
    # Create Eval Data Generator
    eval_DataGen  = DictHiConformerDataGenerator(EvalDatagenConfig, eval = True)
    eval_DataGen.HiC_df = hiconformer_diagonal_builder(len(ref[f"chr{chrom}"][:].seq) // 5000, chrom)
    eval_DataGen._config.BatchSize = 4
    batch_size    = eval_DataGen._config.BatchSize
    NumPred       = len(eval_DataGen.HiC_df)
    print(f"[INFO] Total number interaction pairs: {NumPred}")
    [final_IF,final_ChromHMM_Bin1_states,final_ChromHMM_Bin2_states] = model.predict(eval_DataGen, 
                                                                                    batch_size=None, 
                                                                                    verbose=1, 
                                                                                    steps=None, 
                                                                                    #callbacks=[RegularRender(eval_DataGen, "./MIRTrain", freq = 3)], 
                                                                                    max_queue_size=32,
                                                                                    workers=32, 
                                                                                    use_multiprocessing=False)
    final_IF = final_IF.reshape(-1,1)
    final_ChromHMM_Bin1_states = final_ChromHMM_Bin1_states.reshape(-1, 13)
    final_ChromHMM_Bin2_states = final_ChromHMM_Bin2_states.reshape(-1, 13)
    final_ChromHMM_Bin1_states = [np.argwhere(state>=0.5).reshape(-1) for state in final_ChromHMM_Bin1_states]
    final_ChromHMM_Bin2_states = [np.argwhere(state>=0.5).reshape(-1) for state in final_ChromHMM_Bin2_states]
    target_df = eval_DataGen.HiC_df[:NumPred].copy()
    target_df = target_df[:len(target_df)//(batch_size*100)*(batch_size*100)]
    target_df['Prediction_IF']            = final_IF
    target_df['Prediction_ChromHMM_Bin1'] = final_ChromHMM_Bin1_states
    target_df['Prediction_ChromHMM_Bin2'] = final_ChromHMM_Bin2_states
    target_df = target_df[['chr','bin1','bin2', 'Prediction_IF', 'Prediction_ChromHMM_Bin1','Prediction_ChromHMM_Bin2']]
    target_df.to_csv(f"{OUTPUT_DIR}/Impute_{tissue}_chr{chrom}.csv",index=None)