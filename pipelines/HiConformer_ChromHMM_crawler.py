# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyfaidx import Fasta
from tqdm import tqdm
import pickle
import argparse
import subprocess
import multiprocessing
import copy
import os

## initialize the parser
parser = argparse.ArgumentParser(description='ChromHMM data crawler')
parser.add_argument("--OUTPUT_PATH", help="ChromHMM data path", type=str, default="../data/ChromHMM")
parser.add_argument("--savename", help="save name", type=str, default="IMR90_MboI")
parser.add_argument("--roadmapEID", help="eid", type=str, default="E017")
args = parser.parse_args()
os.makedirs(f"{args.OUTPUT_PATH}/{args.savename}", exist_ok=True)
## download the ChromHMM data
"""
wget https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/imputed12marks/jointModel/final/E
017_25_imputed12marks_hg38lift_stateno.bed.gz -O ChromHMM_25state.bed.gz
gunzip ChromHMM_25state.bed.gz
"""
command = f"wget https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/imputed12marks/jointModel/final/{args.roadmapEID}_25_imputed12marks_hg38lift_stateno.bed.gz -O {args.OUTPUT_PATH}/{args.savename}/ChromHMM_25state.bed.gz"
subprocess.run(command, shell=True)
command = f"gunzip {args.OUTPUT_PATH}/{args.savename}/ChromHMM_25state.bed.gz"
subprocess.run(command, shell=True)

# read the downloaded file
CHROMHMM_PATH  = args.OUTPUT_PATH
CELLTYPE       = args.savename
REF_PATH       = "../data/reference/hg38.fa"
ChromHMM_df = pd.read_csv(f"{CHROMHMM_PATH}/{CELLTYPE}/ChromHMM_25state.bed",sep="\t",header=None)
ChromHMM_df = ChromHMM_df.astype({1: 'int32',2:'int32', 3:'int32'})

# set the reference genome and the chromosome list
Ref = Fasta(REF_PATH, sequence_always_upper=True, read_ahead = 200000000)
chrom_list = [f"chr{i}" for i in range(1,23)]+["chrX"]
ChromHMM_Dict = dict.fromkeys(chrom_list,{})

# ChromHMM_5kb_subprocess_13states
def ChromHMM_5kb_subprocess_13states(chrom,ChromHMM_df,total_len,manager_dict):
    target_df = ChromHMM_df.loc[ChromHMM_df[0]==chrom]
    ChromHMM_Dict = {}
    for idx in tqdm(range(0, total_len, 5000)):   
        label = np.zeros((13))
        within_range = target_df.loc[((idx>=target_df[1])&(target_df[2]>=idx)) |
                                     ((target_df[1]>=idx)&(idx+5000>=target_df[2])) |
                                     ((idx+5000>=target_df[1])&(target_df[2]>=idx+5000))]
        states = [] # Ignore state 25
        for val in within_range[3].values:
            if (val == 1):
                states.append(0)
            elif (val == 2 or val == 3 or val == 4):
                states.append(1)
            elif (val == 5 or val == 6 or val == 7):
                states.append(2)
            elif (val == 8):
                states.append(3)
            elif (val == 9 or val == 10 or val == 11 or val == 12):
                states.append(4)
            elif (val == 13 or val == 14 or val == 15):
                states.append(5)
            elif (val == 16 or val == 17 or val == 18):
                states.append(6)
            elif (val == 19):
                states.append(7)
            elif (val == 25):
                a = 0
            else:
                states.append(val - 12)
        label[states] = 1
        #print(label)
        ChromHMM_Dict[idx] = label
    manager_dict[chrom] = ChromHMM_Dict
    return

# run the merging process
manager = multiprocessing.Manager()
chromHMM_dict = manager.dict()
jobs = []
for chrom in chrom_list:
    p = multiprocessing.Process(target=ChromHMM_5kb_subprocess_13states, args=(chrom,ChromHMM_df,len(Ref[chrom]),chromHMM_dict))
    jobs.append(p)
    p.start()
for proc in jobs:
    proc.join()

ChromHMM_Dict = dict.fromkeys(chrom_list,{})
for chrom in chrom_list:
    ChromHMM_Dict[chrom] = copy.deepcopy(chromHMM_dict[chrom])

# dump the result
fh = open(f"{CHROMHMM_PATH}/{CELLTYPE}/ChromHMM_5kb_13states.pkl", 'wb')
pickle.dump(ChromHMM_Dict, fh)
