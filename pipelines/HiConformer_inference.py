import subprocess
import os
import yaml
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    # use argparse to load all the parameters inference.py needs
    parser = argparse.ArgumentParser(description='Run HiConformer inference on a range of chromosomes for a given tissue and sample.')
    parser.add_argument("--chrom_range", help="Range of chromosome indices to include in the analysis, specified as two numbers (start and end). For example, '1 22' for chromosomes 1 to 22.", type=int, nargs=2, required=True)
    parser.add_argument("--tissue", help="Name of the tissue to be analyzed.", type=str, required=True)
    parser.add_argument("--sample", help="Name of the Avocado sample to be analyzed.", type=str, required=True)
    parser.add_argument("--refseq_path", help="Path to the reference sequence file (e.g., hg38.fa).", type=str, required=True)
    parser.add_argument("--eval_ckpt", help="Path to the pretrained model weights (e.g., ckpt-03440.h5).", type=str, required=True)
    parser.add_argument("--model_config", help="Path to the pretrained model configuration file.", type=str, required=True)
    parser.add_argument("--output_dir", help="Directory path where the inference results will be saved. Default is '../inference_results'.", type=str, default="../inference_results")
    parser.add_argument("--gpu_device", help="GPU device number to use for the analysis. Default is 0.", type=int, default=0)
    args = parser.parse_args()
    chrom_start = args.chrom_range[0]
    chrom_end = args.chrom_range[1]
    print(f"Start inference from {chrom_start} to {chrom_end}")
    for chrom in range(chrom_start, chrom_end + 1):
        print(f"Now inference {args.tissue}-{args.sample}-chr{chrom}")
        process = subprocess.run(["python", "../HiConformer/inference.py",
                                "--chrom", f"{chrom}",
                                "--tissue", f"{args.tissue}",
                                "--sample", f"{args.sample}",
                                "--REF_SEQ", f"{args.refseq_path}",
                                "--EVAL_CKPT", f"{args.eval_ckpt}",
                                "--MODEL_CONFIG", f"{args.model_config}",
                                "--OUTPUT_DIR", f"{args.output_dir}", 
                                "--device", f"{args.gpu_device}"])     
      