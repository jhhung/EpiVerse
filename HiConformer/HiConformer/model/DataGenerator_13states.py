from sklearn.preprocessing import OneHotEncoder
from pyfaidx import Fasta
import pandas as pd
import os
import numpy as np
import pyBigWig as pw
from tqdm import tqdm
import tensorflow as tf
import scipy
from HiConformer.utils.Logger import init_logger,get_logger
import pickle
import random
import gc
import random
from random import sample

RANDOM_STATE = 1832

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DictHiConformerDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, config:dict, eval: bool = False):
        self._config = AttrDict(config)
        self.eval = eval

        ###### Get Logger ######
        self.logger    = get_logger()
        ###### Get Logger ###### 

        ###### Configuration ######
        S = int(self._config.ChromList[0])
        E = int(self._config.ChromList[1])
        self._config.ChromList = [f"chr{X}" for X in range(S,E+1)]
            
        #self.CheckDataExist()
        NumChroms  = len(self._config.ChromList)
        NumTissues = len(self._config.Tissues)
        self.logger.info(f"The Model takes {NumTissues} (Tissues) * {NumChroms} (Chroms) = {NumTissues*NumChroms} Epochs to fully see the entire dataset.")
        ###### Configuration ######

        ###### Sequence ######
        self.Load_Reference_Genome()
        ###### Sequence ######

        ###### Load Hi-C ######
        self.Epochnum    = 0
        self.Tissue_index = list(range(len(self._config["Tissues"])))
        self.LoadTissueData()
        #self.Load_Single_Chrom_HiC_data(self._config.Tissues[0],self._config.ChromList[0])
        ###### Load Hi-C ######
        
        test_data = self.__getitem__(0)
        
        
    def GetTissueAndChromIdx(self):
        Idx        = self.Epochnum // self._config.EpochPerChrom
        Chrom_idx  = Idx % len(self._config.ChromList)
        Tissue_idx = (Idx // len(self._config.ChromList)) % len(self._config.Tissues)
        self.logger.info(f"[Epoch {self.Epochnum}]: Use data {self._config.Tissues[Tissue_idx]}-{self._config.ChromList[Chrom_idx]}")
        return Tissue_idx, Chrom_idx
        
    def CheckDataExist(self):
        self.logger.info("Check all data exist:")
        Missing_Data_Flag = False
        for tissue,sample in zip(self._config.Tissues,self._config.Samples):
            # ChromHMM
            if not os.path.isfile(f"{self._config.ChromHMMPath}/{tissue}/ChromHMM_5kb_13states.pkl"):
                self.logger.warning(f"ChromHMM: {tissue} not exist!")
                Missing_Data_Flag = True
            for chrom in self._config.ChromList:
                # Avocado
                if not os.path.isfile(f"{self._config.AvocadoPath}/{tissue}/{sample}/Archive_{chrom}.npz"):
                    self.logger.warning(f"Avocado: {tissue}-{sample}-{chrom} not exist!")
                    Missing_Data_Flag = True
                # HiC
                if not os.path.isfile(f"{self._config.HiCPath}/{tissue}/{tissue}_{chrom}"):
                    self.logger.warning(f"HiC: {tissue}-{chrom} not exist!")
                    Missing_Data_Flag = True
                # WGBS
                if self._config.UseWGBS:
                    if not os.path.isfile(f"{self._config.WGBSPath}/{tissue}/WGBS_plus_{chrom}.npy"):
                        self.logger.warning(f"WGBS: {tissue}-{chrom} not exist!")
                        Missing_Data_Flag = True
                    if not os.path.isfile(f"{self._config.WGBSPath}/{tissue}/WGBS_minus_{chrom}.npy"):
                        self.logger.warning(f"WGBS: {tissue}-{chrom} not exist!")
                        Missing_Data_Flag = True  
        if Missing_Data_Flag:
            raise FileNotFoundError("Missing data detected!")
            
    def LoadTissueData(self):
        """
        In each epoch, a pair of (tissue,chromosome) is chosen to be trained with.
        """
        Target_tissue_index, Target_chrom_index  = self.GetTissueAndChromIdx()
        Target_tissue = self._config.Tissues[Target_tissue_index]
        Target_sample = self._config.Samples[Target_tissue_index]
        Target_chrom  = self._config.ChromList[Target_chrom_index]

        self.Load_Single_Chrom_HiC_data(Target_tissue,Target_chrom)
        self.Load_Single_Chrom_Sequence_data(Target_chrom)
        self.Load_Single_Chrom_Avocado(Target_tissue,Target_chrom,Target_sample)
        self.Avocado_track = self.Create_Avodict()
        self.query_valid_range()
        self.Load_ChromHMM(Target_tissue,Target_chrom)
                
        if self._config.UseGeneExpression:
            self.Load_Gene_Expression()
        gc.collect()

    def query_valid_range(self):
        seq_min = min(self.ref_seq.keys()); seq_max = max(self.ref_seq.keys())
        avo_min = min(self.Avocado_track.keys()); avo_max = max(self.Avocado_track.keys())
        range_min = max(seq_min,avo_min); range_max = min(seq_max,avo_max)
        # self.HiC_df = self.HiC_df[(self.HiC_df["start1"]>range_min)&(self.HiC_df["start2"]<range_max)]
        self.HiC_df = self.HiC_df[(self.HiC_df["bin1"]>range_min)&(self.HiC_df["bin2"]<range_max)]
        return
                
    def on_epoch_end(self):
        self.Epochnum += 1
        if not self.eval:
            self.logger.info(f"Epoch {self.Epochnum} finished.")
            """
            if self._config.Shuffle:
                TotalEpochs = len(self._config.Tissues) * len(self._config.ChromList)
                if self.Epochnum % TotalEpochs ==0:
                    random.shuffle(self.Tissue_index)
                    random.shuffle(self._config.ChromList)
                    self._config.Tissues = [self._config.Tissues[index] for index in self.Tissue_index]
                    self._config.Samples = [self._config.Samples[index] for index in self.Tissue_index]
                    self.logger.info(f"The training order is shuffled for {self.Epochnum / TotalEpochs} times.")
            """
            if self.Epochnum % self._config.EpochPerChrom == 0:
                self.logger.info(f"On epoch end: len = {self.__len__()}")
                self.LoadTissueData()
                self.logger.info(f"New tissue loaded.")
        
        
    def __len__(self):
        return (len(self.HiC_df)//100)//self._config.BatchSize

    def __getitem__(self, index):
        index = index % self.__len__()
        HiC_Frequency = self.HiC_df[self._config.HiCField][index*self._config.BatchSize*100:(index+1)*self._config.BatchSize*100].values
        HiC_Frequency = HiC_Frequency[np.newaxis].reshape(-1, 100)

        bin1_index    = self.HiC_df['bin1'][index*self._config.BatchSize*100:(index+1)*self._config.BatchSize*100].values
        bin2_index    = self.HiC_df['bin2'][index*self._config.BatchSize*100:(index+1)*self._config.BatchSize*100].values
        
        try:
            Seq_1, Seq_2 = self.Get_Batch_Sequnce(bin1_index, bin2_index)
        except Exception as e:
            self.logger.error(f"Error detected when obtaining batch of sequence! Error msg:{e}")
            self.logger.error(f"Index: {index}, HiC_df length: {len(self.HiC_df)}")
            self.logger.error(f"Query bin1:{bin1_index}")
            self.logger.error(f"Query bin2:{bin2_index}")
            raise e
        try:
            Avo_1, Avo_2 = self.Get_Batch_Avocado(bin1_index, bin2_index)
        except Exception as e:
            self.logger.error(f"Error detected when obtaining batch of Avocado! Error msg:{e}")
            self.logger.error(f"Index: {index}, HiC_df length: {len(self.HiC_df)}")
            self.logger.error(f"Query bin1:{bin1_index}")
            self.logger.error(f"Query bin2:{bin2_index}")
            raise e
        try:
            ChromHMM_bin1 = np.stack([self.ChromHMM[bin1] for bin1 in bin1_index])
            ChromHMM_bin2 = np.stack([self.ChromHMM[bin2] for bin2 in bin2_index])
        except Exception as e:
            self.logger.error(f"Error detected when obtaining batch of ChromHMM! Error msg:{e}")
            self.logger.error(f"Index: {index}, HiC_df length: {len(self.HiC_df)}")
            self.logger.error(f"Query bin1:{bin1_index}")
            self.logger.error(f"Query bin2:{bin2_index}")
            raise e            
            
        dist_weight = self.HiC_df['dist_foldchange'][index*self._config.BatchSize*100:(index+1)*self._config.BatchSize*100].values
        dist_weight = dist_weight[np.newaxis].reshape(-1, 100)
        IF_With_DistWeight = np.stack([HiC_Frequency,dist_weight], axis = 1).astype(np.float32)
        return [Avo_1,Avo_2,Seq_1,Seq_2],\
               [IF_With_DistWeight,ChromHMM_bin1,ChromHMM_bin2] 
                
    def Get_Sample_Weight(self,dist,UseDistPercentile:bool=False):
        if UseDistPercentile:
            Percentile = dist.apply(Percentile_Norm).values
            Percentile = np.expand_dims(Percentile,axis=1)
            return Percentile
        return np.ones((self._config.BatchSize,1))        
                
    def Get_Batch_Sequnce(self,bin1_index:np.ndarray,bin2_index:np.ndarray):
        Seq1 = []; Seq2 = []
        for bin1,bin2 in zip(bin1_index,bin2_index):
            Bin1_seq  = self.ref_seq[bin1]; Bin2_seq  = self.ref_seq[bin2]
            Seq1.append(Bin1_seq); Seq2.append(Bin2_seq)
        Seq1 = np.stack(Seq1,axis=0).reshape(-1, 100, 5000, 4)
        Seq2 = np.stack(Seq2,axis=0).reshape(-1, 100, 5000, 4)
        return Seq1, Seq2    

    def Get_Batch_Avocado(self,bin1_index:np.ndarray,bin2_index:np.ndarray):
        Avocado1 = []; Avocado2 = [];
        for bin1,bin2 in zip(bin1_index,bin2_index):
            Avocado1.append(self.Avocado_track[bin1])             
            Avocado2.append(self.Avocado_track[bin2])
        
        if self._config.AvoChromHMMOnly:
            Avocado1 = np.stack(Avocado1, axis=0).reshape(-1, 100, 200, 13)
            Avocado2 = np.stack(Avocado2, axis=0).reshape(-1, 100, 200, 13) 
        else:
            Avocado1 = np.stack(Avocado1, axis=0).reshape(-1, 100, 200, 71)
            Avocado2 = np.stack(Avocado2, axis=0).reshape(-1, 100, 200, 71) 
        return Avocado1, Avocado2
       
    def Load_Single_Chrom_Avocado(self,Tissue:str,chrom:str,sample:str):
        self.logger.info(f"Now loading Avocado epigenetic tracks")
        self.Avocado_track = None
        self.Avocado_track = np.load(f"{self._config.AvocadoPath}/{Tissue}/{sample}/Archive_{chrom}.npz",mmap_mode='r')['track']
        self.Avocado_track = self.Avocado_track.reshape(-1,84)
        # Remove low quality tracks
        if self._config.UseHighQualityTracks:
            self.logger.info("Using only high-quality Avocado tracks.")
            mask = np.ones(self.Avocado_track.shape[1], dtype=bool)
            Low_quality_index  = [0,3,4,12,40,55,56,74,75,78,79,80,81]
            if self._config.AvoRearrangement:
                ChromHMM_specific_index = [8, 20, 21, 30, 31, 32, 34, 35, 36, 38, 39, 41, 42]
                self.local_ChromHMM = self.Avocado_track[:, ChromHMM_specific_index]
                mask_index = Low_quality_index + ChromHMM_specific_index
                mask[mask_index] = False
                self.Avocado_track = self.Avocado_track[:,mask]
                self.Avocado_track = np.hstack([self.Avocado_track, self.local_ChromHMM])
            else:
                mask[Low_quality_index] = False
                self.Avocado_track = self.Avocado_track[:,mask]
        if self._config.AvoChromHMMOnly:
            self.logger.info("Using ChromHMM Specific Tracks Only!!!!") 
            self.Avocado_track = self.Avocado_track[:, [8, 20, 21, 30, 31, 32, 34, 35, 36, 38, 39, 41, 42]]
        # Use WGBS tracks
        if self._config.UseWGBS:
            self.logger.info("Using WGBS tracks.")
            Len,_       = self.Avocado_track.shape
            Plus_track  = np.load(f"{self._config.WGBSPath}/{Tissue}/WGBS_plus_{chrom}.npy").reshape(-1,1)[:Len,:]
            Minus_track = np.load(f"{self._config.WGBSPath}/{Tissue}/WGBS_plus_{chrom}.npy").reshape(-1,1)[:Len,:]
            self.Avocado_track = np.hstack([self.Avocado_track,Plus_track,Minus_track])
        # Normalization
        if self._config.Normalize=="zscore":
            self.logger.info(f"Now performing z-score transformation for each track")
            self.Avocado_track = scipy.stats.zscore(self.Avocado_track,axis=0,nan_policy='omit')
        elif self._config.Normalize=="MinMax":
            self.logger.info(f"Now performing Minmax transformation for each track")
            self.Avocado_track = (self.Avocado_track-self.Avocado_track.min(axis=0))/self.Avocado_track.max()
        elif self._config.Normalize=="" or self._config.Normalize is None:
            self.logger.info(f"No normalization is applied.")
        else:
            raise NotImplementedError(f"Normalization {self._config.Normalize} is not implemented.")
        #self.Avocado_track = self.Create_Avodict()

    def Create_Avodict(self):
        Avo_dict = {}
        Len, _ = self.Avocado_track.shape
        for i in tqdm(range(5000, Len*25, 5000)):
            Avo_dict[i] = self.Avocado_track[(i//25):(i//25)+200,:]
        return Avo_dict 

    def Load_Reference_Genome(self):
        self.logger.info("Load sequence data")
        self.ref = Fasta(self._config.RefPath, 
                         sequence_always_upper = True, 
                         read_ahead            = 200000000)

    def Load_Single_Chrom_Sequence_data(self,chrom:str):
        self.ref_seq = None
        self.ref_seq = self.ref[chrom][:].seq
        self.ref_seq = self.One_Hot_Encode(self.ref_seq)
        self.ref_seq = self.Create_Seqdict()
        return
        
    def One_Hot_Encode(self,seq:str):
        N_index = [idx for idx,Nuc in enumerate(seq) if Nuc=="N"]
        seq     = seq.replace("N","A")
        mapping = dict(zip("ACGT", range(4)))    
        seq2    = [mapping[i] for i in seq]
        One_hot_matrix = np.eye(4)[seq2]
        # Encode N to [0.25,0.25,0.25,0.25]
        One_hot_matrix[N_index] = np.array([0.25,0.25,0.25,0.25])
        return One_hot_matrix        

    def Create_Seqdict(self):
        Seqdict = {}
        for i in tqdm(range(5000,len(self.ref_seq)+1,5000)):
            Seqdict[i] = self.ref_seq[i:i+5000][:] 
        return Seqdict        

    def get_peakdiag_dfs(self, hic_df: pd.DataFrame, peak_df: pd.DataFrame, neg_ratio: float):
        hic_df = hic_df[['chr', 'bin1', 'bin2', 'Distance', 'dist_foldchange', 'rescaled_intensity']]
        hic_df['Diag_index'] = (hic_df['bin2'] + hic_df['bin1'])//5000
        hic_df['Diag_offset'] = (hic_df['bin2'] - hic_df['bin1'])//5000
        hic_df = hic_df[((hic_df['Diag_offset'] % 2 == 0) & (hic_df['Diag_offset']<=202)) | ((hic_df['Diag_offset'] % 2 != 0) & (hic_df['Diag_offset']<=203))]
        diag_df = hic_df.sort_values(by=['Diag_index', 'Diag_offset']).groupby('Diag_index')
        valid_diag_df = diag_df.filter(lambda x: len(x) == 100)
        diags = valid_diag_df.groupby('Diag_index')
        peak_df['Diag_index'] = (peak_df['bin1'] + peak_df['bin2']) // 5000

        if not self.eval:
            # Sampling
            peak_diag_idxs = [index for index in peak_df['Diag_index'] if index in diags.groups]
            nonpeak_diag_idxs = [index for index in diags.groups if index not in peak_df['Diag_index'].values]

            num_pos_diags = len(peak_diag_idxs)
            if int(num_pos_diags * neg_ratio) > len(nonpeak_diag_idxs):
                num_neg_diags = len(nonpeak_diag_idxs)
            else:
                num_neg_diags = int(num_pos_diags * neg_ratio)
            nonpeak_neg_idxs = sample(nonpeak_diag_idxs, num_neg_diags)
            final_diag_idxs = peak_diag_idxs + nonpeak_neg_idxs
            random.shuffle(final_diag_idxs)

            final_peak_df = [diags.get_group(idx) for idx in final_diag_idxs]
            final_peak_df = pd.concat(final_peak_df)
            return final_peak_df
        else:
            return valid_diag_df
    
    def Load_Single_Chrom_HiC_data(self, tissue: str, chrom:str):
        self.HiC_df = None
        if self._config.UsePeakDiag:
            self.logger.info(f"Use HICCUP peak dataset!")
            self.peak_df = pd.read_csv(f"{self._config.HICCUPPath}/{tissue}/{chrom}-HICCUPS-loops.txt", sep = "\t", usecols = [1, 4], names = ["bin1", "bin2"])
            if not os.path.isfile(f"{self._config.HiCPath}/{tissue}/{tissue}_{chrom}"):
                hic_path = f"{self._config.HiCPath}/{tissue}/{tissue}_{chrom}_sanity"
            else:
                hic_path = f"{self._config.HiCPath}/{tissue}/{tissue}_{chrom}"
            self.HiC_df = pd.read_csv(hic_path, sep = "\t")
            self.HiC_df = self.get_peakdiag_dfs(self.HiC_df, self.peak_df, self._config.NegativeRatio)
        else:
            raise NotImplementedError("Only PeakDiag mode is supported!")
            
        self.logger.info(f"Total available interaction pairs:{len(self.HiC_df)}")
        if self._config.HiCTransform is not None:
            if self._config.HiCTransform == "log10":
                self.logger.info(f"Perform HiC value transformation : {self._config.HiCTransform}")
                self.HiC_df[self._config.HiCField] = np.log10(self.HiC_df[self._config.HiCField])
            elif self._config.HiCTransform == "log2":
                self.logger.info(f"Perform HiC value transformation : {self._config.HiCTransform}")
                self.HiC_df[self._config.HiCField] = np.log2(self.HiC_df[self._config.HiCField])
            elif self._config.HiCTransform == "log":
                self.logger.info(f"Perform HiC value transformation : {self._config.HiCTransform}")
                self.HiC_df[self._config.HiCField] = np.log(self.HiC_df[self._config.HiCField])
                
        # Distance Normalization
        if self._config.DistNorm:
            self.logger.info(f"Perform Distance Normalization")
            # self.HiC_df['Distance'] = (self.HiC_df['start2'] - self.HiC_df['start1'])//5000
            self.HiC_df['Distance'] = (self.HiC_df['bin2'] - self.HiC_df['bin1'])//5000
            if self._config.DistNorm == "log":
                self.logger.info("Log distance is applied.")
                self.HiC_df['Distance'] = np.log(self.HiC_df['Distance'])
            elif self._config.DistNorm == "log10":
                self.logger.info("Log10 distance is applied.")
                self.HiC_df['Distance'] = np.log10(self.HiC_df['Distance'])
            else:
                self.logger.info("Original distance is applied.")
        return

    def Load_ChromHMM(self,Tissue:str,chrom:str):
        self.ChromHMM = None
        self.logger.info(f"Now loading ChromHMM states from {self._config.ChromHMMPath}/{Tissue}")
        fh = open(f"{self._config.ChromHMMPath}/{Tissue}/ChromHMM_5kb_13states.pkl", 'rb')
        self.ChromHMM = pickle.load(fh)
        self.ChromHMM = self.ChromHMM[chrom]

    def Load_Gene_Expression(self,Tissue:str):
        self.GeneExpression = None
        self.logger.info(f"Load {Tissue}'s Gene expression profile.")
        self.GeneExpression = pd.read_csv(f"{self._config.GeneExpressionPath}/{Tissue}/Average.csv")["TPM"].values
        return
