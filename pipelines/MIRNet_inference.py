import subprocess
import os
import yaml
import time
import argparse

if __name__ == "__main__":
    # inference
    # open config and loading general config
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', help='input tissue name')
    parser.add_argument('--input_dir', help='input directory of inference results')
    parser.add_argument('--EVAL_CKPT', help='MIRNet model weights', type=str)
    parser.add_argument('--device', help='GPU device', type=str)
    args = parser.parse_args()
    process = subprocess.run(["python", "../MIRNet/MIRNet_eval.py",
                    "--tissue", f"{args.tissue}",
                    "--EVAL_CKPT", f"{args.EVAL_CKPT}",
                    "--rootfolder", f"{args.input_dir}", 
                    "--device", f"{args.device}"])
    
     
      