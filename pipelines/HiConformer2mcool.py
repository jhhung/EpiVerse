# %%
import cooler
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse
# %%
def init_parser():
    """
    Initialize the parser for the script.
    """
    parser = argparse.ArgumentParser(description='Create Cooler file from 3DIV data. This script processes input 3DIV files and generates a .mcool file with specified resolution for a given tissue type.')
    
    parser.add_argument('--outdir', required=True, help='Output directory path for the resulting .mcool file.')
    parser.add_argument('--inputdir', required=True, help='Input directory path containing 3DIV files.')
    parser.add_argument('-r', '--resolution', type=int, default=5000, help='Resolution for creating the Cooler file. Default is 5000 bp.')
    parser.add_argument('--tissue', required=True, help='Name of the tissue to process.')
    parser.add_argument('--seqpath', required=True, help='Path to the reference sequence file (e.g., hg38.fa).')
    parser.add_argument('--chrom_S', type=int, required=True, help='Starting chromosome number for processing.')
    parser.add_argument('--chrom_E', type=int, required=True, help='Ending chromosome number for processing.')

    return parser.parse_args()

# %%
def read_reference_seq(seq_path):
    """
    Read Reference seq, Default is hg38
    :param: seq_path
    :return: Fasta Object
    """
    from pyfaidx import Fasta
    ref = Fasta(seq_path, sequence_always_upper = True, read_ahead = 200000000)
    return ref

# %%
def main(TISSUE, INPUT_DIR, OUTPUT_DIR, ref, resolution):
    # list contains all bins and pixels information
    bins_all = []
    pixels_all = []
    previous_bins_count = 0
    # print some information
    print(f"Input Dir is: {INPUT_DIR}")
    print(f"Output Dir is: {OUTPUT_DIR}/{TISSUE}")
    os.makedirs(f"{args.outdir}/{TISSUE}", exist_ok=True)
    for i, chr in tqdm(enumerate(chr_names)):
        print(f"Now processing: {TISSUE}-{chr}")
        INPUT_PATH = f"{INPUT_DIR}/{TISSUE}/Impute_{TISSUE}_{chr}.csv"
        print(f"Input file now is: {INPUT_PATH}")   
        # prepare bins df      
        ref_seq = ref[chr][:].seq
        bins = [[chr, i, i + 5000] for i in range(0, len(ref_seq), 5000)]
        print(bins[-10:])
        bins_all += bins
        # prepare pixels df
        convert_data = pd.read_csv(INPUT_PATH) 
        convert_data = convert_data[["bin1", "bin2", "Prediction_IF"]]
        convert_data = convert_data.sort_values(by=["bin1", "bin2"]).reset_index()
        # append pixels df
        if i != 0:
            convert_data[["bin1", "bin2"]] = convert_data[["bin1", "bin2"]] // 5000 + previous_bins_count
            previous_bins_count += len(bins)
        else:
            convert_data[["bin1", "bin2"]] = convert_data[["bin1", "bin2"]] // 5000
            previous_bins_count += len(bins)
        pixels_gt_df = convert_data[["bin1", "bin2", "Prediction_IF"]].set_axis(["bin1_id", "bin2_id", "count"], axis=1)
        #pixels_gt_df = convert_data[["bin1", "bin2", "rescaled_intensity"]].set_axis(["bin1_id", "bin2_id", "count"], axis=1)
        print(f"pixels_df: {pixels_gt_df}")
        pixels_all.append(pixels_gt_df)
    # concat all bins and all pixels across all chromsomes
    print(f"total bins length: {len(bins_all)}")
    bins_df = pd.DataFrame(bins_all, columns=["chrom", "start", "end"])
    print(f"bins df: {bins_df}")
    bins_df.to_csv(f"{OUTPUT_DIR}/{TISSUE}/{TISSUE}_bins.csv", index=None)
    pixels_all_df = pd.concat(pixels_all, axis=0)
    print(f"pixels all df: {pixels_all_df}")
    pixels_all_df.to_csv(f"{OUTPUT_DIR}/{TISSUE}/{TISSUE}_pixels.csv", index=None)
    cooler.create_cooler(f"{OUTPUT_DIR}/{TISSUE}/{TISSUE}.mcool::resolutions/{resolution}", bins=bins_df, pixels=pixels_all_df, dtypes={"count": float})
    
if __name__ == "__main__":
    args = init_parser()
    ref = read_reference_seq(args.seqpath)
    # chr names
    chr_names = ["chr" + str(i) for i in reversed(range(args.chrom_S, args.chrom_E+1))]
    print(chr_names)
    main(args.tissue, args.inputdir, args.outdir, ref, args.resolution)