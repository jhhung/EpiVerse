import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
#from IPython.display import clear_output
from HiConformer.utils.Logger import init_logger,get_logger
import numpy.ma as ma
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import random
import copy
import pandas as pd

import tensorflow_addons as tfa

INTERVAL = 0.85


class RegularRender(tf.keras.callbacks.Callback):
    def __init__(self, eval_datagen, Result_dir, freq=50):
        self.datagen        = eval_datagen
        self.HiC_df         = pd.read_csv("../data/3div/IMR90_MboI/IMR90_MboI_chr17_sanity", sep = "\t")
        self.batch_size     = 4
        self.datagen._config.BatchSize = self.batch_size
        self.Result_dir     = Result_dir
        self.S              = None
        self.E              = None
        self.freq           = freq
        self.cmap_values    = cm.get_cmap('nipy_spectral', 30)
        self.cmap_values    = self.cmap_values(range(3,28,1))
        self.preprocess()
        
    def preprocess(self):
        self.HiC_df         = self.HiC_df[['chr', 'bin1', 'bin2', 'Distance', 'dist_foldchange', 'rescaled_intensity']]
        self.HiC_df['Diag_index'] = (self.HiC_df['bin2'] + self.HiC_df['bin1'])//5000
        self.HiC_df['Diag_offset'] = (self.HiC_df['bin2'] - self.HiC_df['bin1'])//5000
        self.HiC_df = self.HiC_df[((self.HiC_df['Diag_offset'] % 2 == 0) & (self.HiC_df['Diag_offset']<=202)) | ((self.HiC_df['Diag_offset'] % 2 != 0) & (self.HiC_df['Diag_offset']<=203))]        
        
    def get_diag_dfs(self, start:int , end: int):
        target_df = self.HiC_df[self.HiC_df['bin2'].between(start - 1000000, end + 1000000)]
        diag_df = target_df.sort_values(by=['Diag_index', 'Diag_offset']).groupby('Diag_index')
        valid_diag_df = diag_df.filter(lambda x: len(x) == 100)
        return valid_diag_df
        
    def random_sample_region(self):
        while True:
            min_bin1 = self.HiC_df["bin1"].min(); max_bin1 = self.HiC_df["bin1"].max()
            bin1_start = min_bin1 * INTERVAL + max_bin1 * (1 - INTERVAL); bin1_end = min_bin1 * (1 - INTERVAL) + max_bin1 * INTERVAL
            bin1_start = int((bin1_start//5000) * 5000); bin1_end = int((bin1_end//5000) * 5000)
            self.S = random.choice(range(bin1_start, bin1_end, 5000))
            self.E = self.S + 2000000

            print(f"Sampled region: {self.S} - {self.E}")
            target_df = self.get_diag_dfs(self.S, self.E)
            if (len(target_df) <= 10000) or (len(target_df) % (self.batch_size * 100) != 0):
                continue
            else:            
                self.datagen.HiC_df = copy.deepcopy(target_df) # Predict more to avoid 
                print(f"Number of rendered datapoints: {len(self.datagen.HiC_df)}")
                self.NumPred        = len(self.datagen.HiC_df)
                print(f"Number of batches: {len(self.datagen)}")
                break
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.freq==0:
            print(f"Epoch {epoch}: Random sample region to render.")
            self.random_sample_region()
            print(f"Epoch {epoch}: trying to render {self.S}-{self.E} region")
            [final_IF,final_ChromHMM_Bin1_states,final_ChromHMM_Bin2_states] = self.model.predict(self.datagen,
                                                                                                  batch_size = self.batch_size,
                                                                                                  verbose=1, 
                                                                                                  max_queue_size=32,
                                                                                                  workers=32, 
                                                                                                  use_multiprocessing=False)
            final_IF                   = final_IF.reshape(-1,1)
            final_ChromHMM_Bin1_states = final_ChromHMM_Bin1_states.reshape(-1, 13)
            final_ChromHMM_Bin2_states = final_ChromHMM_Bin2_states.reshape(-1, 13)
                       
            print(f"Predicted IF shape: {final_IF.shape}")
            
            Pred_ChromHMM_Bin1 = [np.argwhere(state>=0.5).reshape(-1) for state in final_ChromHMM_Bin1_states]
            Pred_ChromHMM_Bin2 = [np.argwhere(state>=0.5).reshape(-1) for state in final_ChromHMM_Bin2_states]
            target_df          = self.datagen.HiC_df[:self.NumPred].copy()
            print(f"target_df shape: {len(target_df)}")
            GT_ChromHMM_bin1 = [np.argwhere(self.datagen.ChromHMM[bin1]==1).reshape(-1) for bin1 in target_df['bin1'].values]
            GT_ChromHMM_bin2 = [np.argwhere(self.datagen.ChromHMM[bin2]==1).reshape(-1) for bin2 in target_df['bin2'].values]
            target_df['Prediction_IF']            = final_IF[:len(target_df)]
            target_df['Prediction_ChromHMM_Bin1'] = Pred_ChromHMM_Bin1[:len(target_df)]
            target_df['Prediction_ChromHMM_Bin2'] = Pred_ChromHMM_Bin2[:len(target_df)]
            target_df['GT_ChromHMM_start1']       = GT_ChromHMM_bin1[:len(target_df)]
            target_df['GT_ChromHMM_start2']       = GT_ChromHMM_bin2[:len(target_df)]

            self.Visualize_Contact_map(target_df,self.S,self.E,epoch)
        
    def get_state_array(self,states):
        array  = np.zeros((13))
        array[states] = 1
        return array

    def Visualize_Contact_map(self,df,Start,End,epoch):
        total_bins    = int((End-Start) / 5000) + 1
        print(f"Total bins: {total_bins}")
        
        # Contact map
        original_cmap = np.zeros((total_bins, total_bins),dtype=float)  
        Predict_cmap  = np.zeros((total_bins, total_bins),dtype=float)

        target_df = df.loc[((df["bin1"]>=Start)&(df["bin1"]<=End)) & ((df["bin2"]>=Start)&(df["bin2"]<=End))]
        target_df.sort_values(by=["bin1","bin2"],inplace=True,ignore_index=True)
        row_index = ((target_df['bin1'].values-Start)//5000).astype("int32")
        col_index = ((target_df['bin2'].values-Start)//5000).astype("int32")
        
        original_cmap[row_index,col_index] = target_df["rescaled_intensity"]
        original_cmap = original_cmap + original_cmap.T - np.diag(np.diag(original_cmap))
        Predict_cmap[row_index,col_index] = target_df['Prediction_IF']
        Predict_cmap = Predict_cmap + Predict_cmap.T - np.diag(np.diag(Predict_cmap))
        
        N = len(target_df)
        
        try:
            GT_bin1_states   = [self.get_state_array(state) for state in target_df["GT_ChromHMM_start1"].values]
            GT_bin2_states   = [self.get_state_array(state) for state in target_df["GT_ChromHMM_start2"].values]
            Pred_bin1_states = [self.get_state_array(state) for state in target_df["Prediction_ChromHMM_Bin1"].values]
            Pred_bin2_states = [self.get_state_array(state) for state in target_df["Prediction_ChromHMM_Bin2"].values]  

            # ChromHMM

            GT_bin1_ChromHMM_matrix = np.zeros((total_bins,13))
            GT_bin1_ChromHMM_matrix[row_index,:]   = GT_bin1_states
            GT_bin2_ChromHMM_matrix = np.zeros((total_bins,13))
            GT_bin2_ChromHMM_matrix[col_index,:]   = GT_bin2_states
            Pred_bin1_ChromHMM_matrix = np.zeros((total_bins,13))
            Pred_bin1_ChromHMM_matrix[row_index,:] = Pred_bin1_states
            Pred_bin2_ChromHMM_matrix = np.zeros((total_bins,13))
            Pred_bin2_ChromHMM_matrix[col_index,:] = Pred_bin2_states
            
            fig = plt.figure(figsize = (64,32))
            gs  = fig.add_gridspec(2, 4,  width_ratios=(4,28,4,28), height_ratios=(4,28),
                                   left=0.05, right=0.95, bottom=0.05, top=0.95,
                                   wspace=0.05, hspace=0.05)

            # PLOT
            ## GT
            ax_gt_ChromHMM_bin1 = fig.add_subplot(gs[1, 0])
            ax_gt_ChromHMM_bin2 = fig.add_subplot(gs[0, 1])
            ax_gt_Contact_map   = fig.add_subplot(gs[1, 1])

            for col in range(13):
                mask = np.ones(GT_bin1_ChromHMM_matrix.shape)
                mask[:,col] = False
                mx = ma.masked_array(GT_bin1_ChromHMM_matrix, mask=mask)
                colors = [(0,[0,0,0,1]),(1,self.cmap_values[col])]
                new_cmap = LinearSegmentedColormap.from_list("Custom_cmap",colors,N=2)    
                ax_gt_ChromHMM_bin1.matshow(mx,cmap=new_cmap)

            for col in range(13):
                mask = np.ones(GT_bin2_ChromHMM_matrix.shape)
                mask[:,col] = False
                mx = ma.masked_array(GT_bin2_ChromHMM_matrix, mask=mask)
                colors = [(0,[0,0,0,1]),(1,self.cmap_values[col])]
                new_cmap = LinearSegmentedColormap.from_list("Custom_cmap",colors,N=2)
                ax_gt_ChromHMM_bin2.matshow(mx.T,cmap=new_cmap)
            ax_gt_Contact_map.matshow(original_cmap,cmap="YlOrRd",vmax=10)

            ## Pred
            ax_pred_ChromHMM_bin1 = fig.add_subplot(gs[1, 2])
            ax_pred_ChromHMM_bin2 = fig.add_subplot(gs[0, 3])
            ax_pred_Contact_map   = fig.add_subplot(gs[1, 3])

            for col in range(13):
                mask = np.ones(Pred_bin1_ChromHMM_matrix.shape)
                mask[:,col] = False
                mx = ma.masked_array(Pred_bin1_ChromHMM_matrix, mask=mask)
                colors = [(0,[0,0,0,1]),(1,self.cmap_values[col])]
                new_cmap = LinearSegmentedColormap.from_list("Custom_cmap",colors,N=2)    
                ax_pred_ChromHMM_bin1.matshow(mx,cmap=new_cmap)

            for col in range(13):
                mask = np.ones(Pred_bin2_ChromHMM_matrix.shape)
                mask[:,col] = False
                mx = ma.masked_array(Pred_bin2_ChromHMM_matrix, mask=mask)
                colors = [(0,[0,0,0,1]),(1,self.cmap_values[col])]
                new_cmap = LinearSegmentedColormap.from_list("Custom_cmap",colors,N=2)
                ax_pred_ChromHMM_bin2.matshow(mx.T,cmap=new_cmap)
            correlation = target_df['Prediction_IF'].corr(target_df['rescaled_intensity'])
            ax_pred_Contact_map.matshow(Predict_cmap,cmap="YlOrRd",vmax=10)
            ax_pred_Contact_map.text(200, 15, f"Epoch {epoch}: corr={correlation}", bbox={'facecolor': 'white', 'pad': 10},fontsize=30)
            plt.title(f"Region: {Start}-{End}", fontsize = 50)
            fig.savefig(f"{self.Result_dir}/contactmaps/{self.datagen._config.ChromList[0]}-epoch_{epoch}.png")
            
            
        except:
            print("ChromHMM plot error!")
            print(f"Row index: {row_index}")
            print(f"Col index: {col_index}")
            print(f"GT bin1 states: {GT_bin1_states}")
            print(f"GT bin2 states: {GT_bin2_states}")
    
    
class PlotLearning(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        #clear_output(wait=True)
        
        ax1.title.set_text('Loss')
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        
        ax2.title.set_text('Validation')        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show()

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.

    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.

    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.

    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.

    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []
        self.logger  = get_logger()
        
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if batch % 2048 == 0:
            self.logger.info('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))

class WeightAdjuster(tf.keras.callbacks.Callback):
    def __init__(self, weights: list, total_steps: int,logger=None):
        self.interaction_weight  = weights["Interaction_Frequency"]
        self.bin1_IS_weight      = weights["Bin1_ChromHMM_State"]
        self.bin2_IS_weight      = weights["Bin2_ChromHMM_State"]
        self.total_steps         = total_steps
        self.logger              = get_logger()
    def on_epoch_end(self, epoch, logs={}):
        new_alpha = self.bin1_IS_weight * np.cos(np.pi * epoch / float(2*self.total_steps))
        if new_alpha < 0.2:
            new_alpha = 0.2
        # Updated loss weights
        K.set_value(self.interaction_weight , 1-new_alpha)
        K.set_value(self.bin1_IS_weight, new_alpha)
        K.set_value(self.bin2_IS_weight, new_alpha)
        if self.logger:
            self.logger.info(f"Epoch {epoch}: ChromHMM weight = {self.bin1_IS_weight}, Interaction weight = {self.interaction_weight}")
