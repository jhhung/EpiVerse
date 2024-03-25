import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
import tensorflow_addons as tfa


def Percentile_Norm(val):
    if val<1000000*0.2:
        return 0.4
    elif val<1000000*0.4:
        return 0.4
    elif val<1000000*0.5:
        return 0.45
    elif val<1000000*0.6:
        return 0.45
    elif val<1000000*0.7:
        return 0.5
    elif val<1000000*0.8:
        return 0.5
    elif val<1000000*0.85:
        return 0.55
    elif val<1000000*0.90:
        return 0.55
    elif val<1000000*0.95:
        return 0.6
    else:
        return 0.6

def weighted_pearson_r(y_true, y_pred):
    x      = y_pred[:,:,0]
    y      = y_true[:,0,:]
    weight = y_true[:,1,:]
    
    mx = K.sum(x*weight, axis = 1) / K.sum(weight)
    my = K.sum(y*weight, axis = 1) / K.sum(weight)
    
    mx = tf.expand_dims(mx, axis=1)
    my = tf.expand_dims(my, axis=1)

    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym * weight, axis = 1)
    x_square_sum = K.sum(xm * xm * weight, axis = 1)
    y_square_sum = K.sum(ym * ym * weight, axis = 1)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def weighted_distance_pearson_r(y_true, y_pred):
    x      = y_pred[:,:,0]
    y      = y_true[:,0,:]
    weight = y_true[:,1,:]
    
    mx = K.sum(x * weight, axis = 0) / K.sum(weight, axis = 0)
    my = K.sum(y * weight, axis = 0) / K.sum(weight, axis = 0)
    
    xm = x - mx
    ym = y - my
    r_num = K.sum(xm * ym * weight, axis = 0)
    x_square_sum = K.sum(xm * xm * weight, axis = 0)
    y_square_sum = K.sum(ym * ym * weight, axis = 0)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def weighted_cosine_similarity(y_true, y_pred):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
    """
    x      = y_pred[0]
#    y_true = tf.expand_dims(y_true,axis=-1)
    y      = y_true[0]
    weight = y_true[1]
    
    numerator    = K.sum(weight*x*y, axis=0)
        
    x_square_sum = K.sum(x * x * weight)
    y_square_sum = K.sum(y * y * weight)
    x_sqrt       = K.sqrt(x_square_sum)
    y_sqrt       = K.sqrt(y_square_sum)

    denominator  = x_sqrt * y_sqrt
    return numerator/denominator

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig

def Shrinkage_loss(a=10,c=0.2):
    def loss(y_true, y_pred):
        residual  = tf.keras.backend.abs(y_pred - y_true)
        numerator = tf.keras.backend.square(residual)  # (batch_size, 2)
        denominator = 1 + tf.keras.backend.exp(a*(c-residual))
        return numerator / denominator
    return loss
    
def Focal_RLoss(beta=0.5, gamma=2.0):
    """Reference: https://arxiv.org/pdf/2102.09554.pdf, regression version of focal loss"""
    def loss(y_true, y_pred):
        Diff = y_true-y_pred
        Loss = (sigmoid(tf.math.abs(Diff*beta))**gamma) * Diff
        return tf.reduce_mean(Loss)
    return loss      
    
def L1_L2_Distance_Corr_Loss(W_l1):
    def weighted_L1(y_true, y_pred, weight):
        return K.sum(K.abs(weight * (y_true - y_pred)), axis = 1)
    def weighted_L2(y_true, y_pred, weight):
        return K.sum(weight * K.square(y_true - y_pred), axis = 1)
    def loss(y_true, y_pred):
        weighted_corr = weighted_pearson_r(y_true, y_pred)
        weighted_dist_corr = weighted_distance_pearson_r(y_true, y_pred)
        y_pred = y_pred[:,:,0]
        gt     = y_true[:,0,:]
        weight = y_true[:,1,:]
        #test(gt[:2],y_pred[:2],weight[:2])
        weighted_L1_Loss = weighted_L1(gt,y_pred,weight)
        weighted_L2_Loss = weighted_L2(gt,y_pred,weight)
        L1_L2  = K.mean(W_l1 * weighted_L1_Loss+ (1 - W_l1) * weighted_L2_Loss)
        return L1_L2 + 5.0 * (1 - weighted_corr) + 100 * (1 - weighted_dist_corr)
    def test(y_true,y_pred, weight):
        K.print_tensor(y_true, message='GT = ')
        K.print_tensor(weight, message='Weight = ')
        K.print_tensor(y_pred, message='Pred = ')
        K.print_tensor(weighted_L1(y_true,y_pred,weight), message='L1 = ')
        K.print_tensor(weighted_L2(y_true,y_pred,weight), message='L2 = ')        
    return loss

def L1_L2_Corr_Loss(W_l1):
    def weighted_L1(y_true, y_pred, weight):
        return K.sum(K.abs(weight * (y_true - y_pred)), axis = 1)
    def weighted_L2(y_true, y_pred, weight):
        return K.sum(weight * K.square(y_true - y_pred), axis = 1)
    def loss(y_true, y_pred):
        weighted_corr = weighted_pearson_r(y_true, y_pred)
        weighted_dist_corr = weighted_distance_pearson_r(y_true, y_pred)
        y_pred = y_pred[:,:,0]
        gt     = y_true[:,0,:]
        weight = y_true[:,1,:]
        #test(gt[:2],y_pred[:2],weight[:2])
        weighted_L1_Loss = weighted_L1(gt,y_pred,weight)
        weighted_L2_Loss = weighted_L2(gt,y_pred,weight)
        L1_L2  = K.mean(W_l1 * weighted_L1_Loss+ (1 - W_l1) * weighted_L2_Loss)
        return L1_L2 + 5.0 * (1 - weighted_corr)
    def test(y_true,y_pred, weight):
        K.print_tensor(y_true, message='GT = ')
        K.print_tensor(weight, message='Weight = ')
        K.print_tensor(y_pred, message='Pred = ')
        K.print_tensor(weighted_L1(y_true,y_pred,weight), message='L1 = ')
        K.print_tensor(weighted_L2(y_true,y_pred,weight), message='L2 = ')       
    return loss
     
def multilabel_focal_loss(alpha, epsilon = 1.e-10, gamma = 2.0):
    """
    alpha : Control Pos/Neg balance. When alpha is large, pos data is heavy weighted
    gamma : Control easy case weighting. When gamma is large, easy cases is light-weighted
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    def multilabel_focal_loss_fn(y_true, y_pred):
        y_pred     = tf.reshape(y_pred, [-1, 13])
        y_true     = tf.cast(y_true, tf.float32)
        y_false    = 1 - y_true
        
        y_pred     = tf.clip_by_value(y_pred, epsilon, 1. - epsilon) # Clip then log to avoid nan problems
        y_pred     = tf.cast(y_pred, tf.float32)
        pos_log_pred = tf.math.log(y_pred)
        neg_log_pred = tf.math.log(1-y_pred) 
        
        pos_weight   = tf.math.pow(tf.subtract(1., y_pred), gamma)      
        neg_weight   = tf.math.pow(y_pred, gamma)  
                
        pos_CE       = tf.multiply(tf.multiply(tf.multiply(pos_weight, pos_log_pred), alpha),y_true)
        neg_CE       = tf.multiply(tf.multiply(neg_weight, neg_log_pred),y_false)

        fl   = -(pos_CE+neg_CE)
        loss = tf.reduce_mean(fl)
        return loss
    return multilabel_focal_loss_fn

class F1_Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.f1_score_fn = tfa.metrics.F1Score(num_classes = 13, average = "weighted", threshold = 0.5)

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.reshape(y_true, (-1, 13))
        y_pred = tf.reshape(y_pred, (-1, 13))
        f1_score = self.f1_score_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(f1_score)

    def result(self):
        return self.f1

    def reset_states(self):
        self.f1_score_fn.reset_states()
        self.f1.assign(0)
