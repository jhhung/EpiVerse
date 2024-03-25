# %%
import cooler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm 
import fanc
import argparse

def init_parser():
    """
    HiConformer parser
    """
    parser = argparse.ArgumentParser(description='MIRNetDataGen')
    parser.add_argument('--chrom_S', help='input chrom start', type=int)
    parser.add_argument('--chrom_E', help='input chrom end', type=int)
    parser.add_argument('--tissue', help="input tissue name", type=str)
    parser.add_argument('--OUTPUT_DIR', help="output dir", type=str)
    parser.add_argument('--imputation', help="imputation mcool path", type=str)
    parser.add_argument('--gt', help="ground truth mcool path", type=str)
    parser.add_argument('--mode', help="mode", type=str)
    args = parser.parse_args()
    return args
# %%
if __name__ == "__main__":
    parser = init_parser()
    tissue = parser.tissue
    GT_PATH = parser.gt
    #IMPUTATION_PATH = f"{parser.OUTPUT_DIR}/{parser.tissue}/{parser.tissue}.mcool@5000"
    IMPUTATION_PATH = parser.imputation
    OUT_PATH = f"{parser.OUTPUT_DIR}/{parser.tissue}/HiConformer2MIRData"

    if parser.mode == "imputation_mode":
        GT_PATH = IMPUTATION_PATH
    gt = fanc.load(GT_PATH)
    pred = fanc.load(IMPUTATION_PATH)
    chr_names = ["chr" + str(i) for i in reversed(range(parser.chrom_S, parser.chrom_E + 1))]
    for chr in chr_names:
        gt_matrix = gt.matrix((chr, chr))
        pred_matrix = pred.matrix((chr, chr))
        print(f"Now processing {tissue}-{chr}")
        os.makedirs(f"{OUT_PATH}/imputation/figures", exist_ok=True)
        os.makedirs(f"{OUT_PATH}/ground_truth/figures", exist_ok=True)
        for i in tqdm(range(400, len(gt_matrix), 50)):
            if (i + 400 < len(gt_matrix)):
                gt_part = np.array(gt_matrix[i:i+200, i:i+200])
                pred_part = np.array(pred_matrix[i:i+200, i:i+200])
                if (np.count_nonzero(gt_part) and np.count_nonzero(pred_part)):
                    np.save(f"{OUT_PATH}/ground_truth/{chr}-{i*5000}-{(i+200)*5000}.npy", gt_part)
                    np.save(f"{OUT_PATH}/imputation/{chr}-{i*5000}-{(i+200)*5000}.npy", pred_part)
                    plt.matshow(gt_part, cmap="YlOrRd", vmax=10)
                    plt.savefig(f"{OUT_PATH}/ground_truth/figures/{chr}-{i*5000}-{(i+200)*5000}.png")
                    plt.close()
                    plt.matshow(pred_part, cmap="YlOrRd", vmax=10)
                    plt.savefig(f"{OUT_PATH}/imputation/figures/{chr}-{i*5000}-{(i+200)*5000}.png")
                    plt.close()
