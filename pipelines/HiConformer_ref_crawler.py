import subprocess
import argparse
import os
"""
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa
"""
# store the path of the reference genome
parser = argparse.ArgumentParser(description='Reference data crawler')
parser.add_argument("--OUTPUT_PATH", help="output data path", type=str, default="../data/reference")
args = parser.parse_args()
os.makedirs(args.OUTPUT_PATH, exist_ok=True)
command = f"wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -O {args.OUTPUT_PATH}/hg38.fa.gz"
subprocess.run(command, shell=True)
command = f"gunzip {args.OUTPUT_PATH}/hg38.fa.gz"
subprocess.run(command, shell=True)
