import subprocess
import os
import yaml
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="imputation_mode", help="Mode for HiConformer2MIRNet. Default is imputation_mode. Other mode is ground_truth_mode.")
    parser.add_argument("--chrom_S", type=int, default=1, help="Starting chromosome number for processing.")
    parser.add_argument("--chrom_E", type=int, default=22, help="Ending chromosome number for processing.")
    parser.add_argument("--tissue", type=str, default="IMR90_MboI", help="Name of the tissue to process.")
    parser.add_argument("--gt", type=str, default=" ", help="Path to the ground truth .mcool file (Only for ground truth mode). Default is ' '.")
    parser.add_argument("--outdir", type=str, default="../inference_results", help="Output directory for inference results. Default is ../inference_results.")
    parser.add_argument("--imputation", help="imputation mcool path", type=str)
    args = parser.parse_args()
    process = subprocess.run(["python", "../MIRNet/MIRDataGen.py",
                                "--chrom_S", f"{args.chrom_S}",
                                "--chrom_E", f"{args.chrom_E}",
                                "--tissue", f"{args.tissue}",
                                "--gt", f"{args.gt}",
                                "--mode", f"{args.mode}",
                                "--OUTPUT_DIR", f"{args.outdir}",
                                "--imputation", f"{args.imputation}"])
                                
     
      