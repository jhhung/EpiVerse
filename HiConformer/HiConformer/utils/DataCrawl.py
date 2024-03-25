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


##### AVOCADO #####

AVOCADO_assay_list = ['ATAC-seq'           ,'ATF3'              ,'BHLHE40'                     ,'CAGE_minus'                 ,
                      'CAGE_plus'          ,'CEBPB'             ,'CHD2'                        ,'CTCF'                       , 
                      'DNase'              ,'EGR1'              ,'ELF1'                        ,'ELK1'                       ,
                      'EP300'              ,'ETS1'              ,'EZH2'                        ,'EZH2phosphoT487'            , 
                      'FOXA1'              ,'FOXK2'             ,'GABPA'                       ,'GTF2F1'                     , 
                      'H2AFZ'              ,'H2AK5ac'           ,'H2BK120ac'                   ,'H2BK12ac'                   , 
                      'H2BK15ac'           ,'H2BK5ac'           ,'H3F3A'                       ,'H3K14ac'                    , 
                      'H3K18ac'            ,'H3K23ac'           ,'H3K27ac'                     ,'H3K27me3'                   , 
                      'H3K36me3'           ,'H3K4ac'            ,'H3K4me1'                     ,'H3K4me2'                    , 
                      'H3K4me3'            ,'H3K79me1'          ,'H3K79me2'                    ,'H3K9ac'                     ,
                      'H3K9me2'            ,'H3K9me3'           ,'H4K20me1'                    ,'H4K8ac'                     , 
                      'H4K91ac'            ,'HDAC2'             ,'JUND'                        ,'KDM1A'                      , 
                      'MAFK'               ,'MAX'               ,'MAZ'                         ,'NRF1'                       ,
                      'POLR2A'             ,'POLR2AphosphoS5'   ,'RAD21'                       ,'RAMPAGE_minus'              , 
                      'RAMPAGE_plus'       ,'RCOR1'             ,'REST'                        ,'RFX5'                       , 
                      'RXRA'               ,'SIN3A'             ,'SMC3'                        ,'SP1'                        , 
                      'SUZ12'              ,'TAF1'              ,'TARDBP'                      ,'TBP'                        , 
                      'TCF12'              ,'USF1'              ,'USF2'                        ,'YY1'                        ,
                      'ZBTB33'             ,'ZFP36'             ,'microRNA-seq_minus'          ,'microRNA-seq_plus'          , 
                      'polyA_RNA-seq_minus','polyA_RNA-seq_plus','polyA_depleted_RNA-seq_minus','polyA_depleted_RNA-seq_plus', 
                      'small_RNA-seq_minus','small_RNA-seq_plus','total_RNA-seq_minus'         ,'total_RNA-seq_plus'          ]


def Crawl_Avocado_assays(Sample,metadata,Storage_path):
    """[summary]

    Args:
        Sample ([type]): [description]
        metadata ([type]): [description]
        Storage_path ([type]): [description]
    """

    target_df     = metadata.loc[metadata['Sample']==Sample]
    target_df     = target_df.reset_index()
    target_tissue = target_df['Biosample term name'][0] 

    Saved_path = f"{Storage_path}/{target_tissue}/{Sample}/"

    if not os.path.isdir(f"{Saved_path}"):
        os.makedirs(f"{Saved_path}")

    if os.listdir(f"{Saved_path}"):
        result = input(f"There's already files inside {Saved_path}, still download?(y/n)")
        if result == "n":
            return

    print(f"Now crawling {Sample},{target_tissue} from ENCODE Project.")

    for assay_idx in tqdm.tqdm(range(len(target_df))):
        download_url = f"https://www.encodeproject.org{target_df['File download URL'][assay_idx]}"        
        target_assay = target_df['Assay'][assay_idx]
    
        print(f"Downloading {target_assay}")

            
        wget.download(download_url, out = f"{Storage_path}/{target_tissue}/{Sample}/{target_assay}.bigwig")
    
    print("Md5sum sanity check:")
    md5sumcheck(Sample,metadata,Storage_path)    
    return

def md5sumcheck(sample,df,Storage_path):
    """[summary]

    Args:
        AVOCADO_assay_list ([type]): [description]
        tissue ([type]): [description]
        sample ([type]): [description]
        df ([type]): [description]
        Storage_path ([type]): [description]
    """
    target_df     = df.loc[df["Sample"]==sample].reset_index()
    target_tissue = target_df['Biosample term name'][0] 

    for assay in tqdm.tqdm(AVOCADO_assay_list):
        if os.path.isfile(f"{Storage_path}/{target_tissue}/{sample}/{assay}.bigwig"):
            continue
        else:
            print(assay+" doesn't exist!!")

    # Mulit-processing
    cpus = mp.cpu_count()
    pool = mp.Pool(processes=cpus)

    result=[]

    results = [pool.apply_async(checksum_subprocess, (df_slice.reset_index(),sample,Storage_path))for df_slice in np.array_split(target_df, cpus)]
    pool.close()
    pool.join()


    for result in results:
        print(result.get())
    return
    
def checksum_subprocess(df,sample,Storage_path):
    """[summary]

    Args:
        df ([type]): [description]
        sample ([type]): [description]
    """
    tissue    = df['Biosample term name'][0]

    for i in tqdm.tqdm(range(len(df))):
        
        gt_md5sum = df['md5sum'][i]
        target_Assay = df['Assay'][i]
        target_tissue = df['Biosample term name'][0] 

        data = f"{Storage_path}/{target_tissue}/{sample}/{target_Assay}.bigwig"
                            
        target_md5sum= subprocess.run(['md5sum', data], stdout=subprocess.PIPE)
        target_md5sum= target_md5sum.stdout.decode('utf-8').split(" ")[0]
                
        if target_md5sum!= gt_md5sum:
            print(target_Assay, "for ",sample," is not correct!")    
    return

def Get_Avocado_Celltypes(metadata):
    df = pd.read_csv(metadata)
    celltypes = df['Biosample term name'].unique()    
    return df,celltypes

def Get_Avocado_SampleName(df,celltype):
    if celltype.capitalize() in df['Biosample term name'].unique():
        target_celltype = celltype.capitalize()
    elif celltype in df['Biosample term name'].unique():
        target_celltype = celltype
    else:
        print("Target tissue is not including in Avocado!")
        return
    return df.loc[df['Biosample term name']==target_celltype]['Sample'].unique()

def Get_Avocado_Archive(AVOCADO_PATH,chrom,celltype,sample):
    
    bin1_tracks = []
    bin2_tracks = []
    data        = []
    
    assay_tracks = [pyBigWig.open(f"{AVOCADO_PATH}/{celltype}/{sample}/{assay}.bigwig") for assay in AVOCADO_assay_list]
    
    # Cache data
    for idx,track in tqdm.tqdm(enumerate(assay_tracks)):
        print(AVOCADO_assay_list[idx])
        chromosome_length = track.chroms(chrom)
        signal            = track.values(chrom,0,chromosome_length)[0::25]
        track.close()
        data.append(signal)
        
    data              = np.column_stack(data)
    num_loci, _       = data.shape
    Nearest_porpotion = (num_loci//200)*200
    data              = data[:Nearest_porpotion,:]
    data              = data.reshape(-1,200,84)
    
    np.savez(f"{AVOCADO_PATH}/{celltype}/{sample}/Archive_{chrom}.npz", track=data)
    return

##### AVOCADO #####

##### 3DIV #####
def Crawl_3DIV_celltype(Storage_path,celltype):
    
    Saved_path = f"{Storage_path}/{celltype}"

    if len(os.listdir(f"{Saved_path}/")) != 0:
        result = input(f"There's already files inside {Saved_path}, still download?(y/n)")
        if result == "n":
            return

    all_chrom = [f"chr{i}" for i in range(1,23)]
    all_chrom.append("chrX")
    
    for chrom in tqdm.tqdm(all_chrom):
        command = f"curl ftp://ftp_3div:3div@ftp.kobic.re.kr/Normal_Hi-C\(hg38\)/{celltype}/{celltype}.{chrom}.distnorm.scale2.gz --user ftp_3div:3div -o {Saved_path}/{celltype}_{chrom}.gz"
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




