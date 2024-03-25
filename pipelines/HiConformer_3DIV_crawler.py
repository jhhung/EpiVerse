# %%
import os
import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import subprocess
import pyBigWig
import numpy as np
import urllib
import multiprocessing as mp
from sklearn.preprocessing import normalize
import subprocess
import sys
import wget
from pyfaidx import Fasta
import argparse

##### 3DIV #####
def Crawl_3DIV_celltype(Storage_path,celltype,savename):
    
    Saved_path = f"{Storage_path}/{savename}"
    os.makedirs(Saved_path,exist_ok=True)
    if len(os.listdir(f"{Saved_path}/")) != 0:
        result = input(f"There's already files inside {Saved_path}, still download?(y/n)")
        if result == "n":
            return

    all_chrom = [f"chr{i}" for i in range(1,23)]
    all_chrom.append("chrX")
    
    for chrom in tqdm.tqdm(all_chrom):
        command = f"curl ftp://ftp_3div:3div@ftp.kobic.re.kr/Normal_Hi-C\(hg38\)/{celltype}/{celltype}.{chrom}.distnorm.scale2.gz --user ftp_3div:3div -o {Saved_path}/{savename}_{chrom}.gz"
        try:
            output = subprocess.check_output(command, shell=True)
        except subprocess.CalledProcessError as e:
            output = e.output
            print(output)
    
    #Unzip all
    command = f"gunzip {Saved_path}/*.gz"
    result = subprocess.check_output(command, shell=True)
    return

def collect_results(result):
    results.extend(result)

def sanity_check(chrom,df,ref_path,celltype,HIC_PATH):

    # Mulit-processing
    cpus = mp.cpu_count()
    pool = mp.Pool(processes=cpus)
    print(cpus)
    results = [pool.apply_async(sanity_check_subprocess, (df_slice.reset_index(),chrom,ref_path,)) for df_slice in np.array_split(df, cpus)]

    pool.close()
    pool.join()

    new_df = pd.concat([result.get() for result in results])
    new_df.to_csv(f"{HIC_PATH}/{celltype}/{celltype}_{chrom}_sanity",sep='\t')
    return


def sanity_check_subprocess(df,chrom,ref_path):

    ref = Fasta(ref_path)

    Unknown_nuc_index = []
    with tqdm.tqdm(total=len(df)) as pbar: 
        for index,row in df.iterrows():
            Bin1_seq = ref[chrom][row['bin1']:row['bin1']+5000].seq
            Bin2_seq = ref[chrom][row['bin2']:row['bin2']+5000].seq
            if "N" in Bin1_seq.upper() or "N" in Bin2_seq.upper():
                Unknown_nuc_index.append(index)
            pbar.update(1)
    df = df.drop(Unknown_nuc_index)
    return df        
##### 3DIV #####
# %%
parser = argparse.ArgumentParser(description='3DIV data crawler')
parser.add_argument("--OUTPUT_PATH", help="output data path", type=str, default="../data/3div")
parser.add_argument("--threeDIVname", help="3DIVname", type=str, default="IMR90_MboI")
parser.add_argument("--savename", help="saveas", type=str, default="IMR90_MboI")
args = parser.parse_args()
Crawl_3DIV_celltype(args.OUTPUT_PATH, args.threeDIVname, args.savename)
