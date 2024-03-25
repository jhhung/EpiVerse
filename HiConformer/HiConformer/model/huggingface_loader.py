import tensorflow as tf
import numpy as np
from HiConformer.utils.Logger import init_logger,get_logger

def model_selector(config:dict):
    logger = get_logger()
    logger.info("[HuggingFace] Use HuggingFace transformer models.")
    """
    For bert, no embedding_size, only hidden_size. Hidden_size = 2 * bin_filters
    For AlBert, embedding_size = 2 * bin_filters
    """
    if config["model"] == "Bert":
        logger.info("TFBert is applied.")
        from transformers import TFBertModel as HuggingFaceModel
        from transformers import BertConfig  as HuggingFaceConfig
        Config = HuggingFaceConfig(
                                   vocab_size          = 1, # Not used, since embeddings comes from feature extractor
                                   hidden_size         = config["EmbedDim"], 
                                   num_hidden_layers   = config["NumberOfBlocks"],
                                   num_attention_heads = config["NumberOfHeads"],
                                   intermediate_size   = config["FFdims"],
                                   hidden_act          = config["Activation"],
                                   hidden_dropout_prob          = 0.1,
                                   attention_probs_dropout_prob = 0.1,
                                   max_position_embeddings      = 512,
                                   type_vocab_size              = 3, # Seq1, window, seq2
                                   initializer_range            = 0.02,
                                   layer_norm_eps               = 1e-12,
                                   pad_token_id                 = 0,
                                   position_embedding_type      = 'relative_key', # absolute, or relative_key
                                   output_attentions = True,
                                   use_cache   = True,
                                   return_dict = True,
                                   training    = True)
    elif config["model"] == "ALBert":
        from transformers import TFAlbertModel as HuggingFaceModel
        from transformers import AlbertConfig  as HuggingFaceConfig
        Config = HuggingFaceConfig(
                                   vocab_size          = 1, # Not used, since embeddings comes from feature extractor
                                   embedding_size      = config["EmbedDim"],
                                   hidden_size         = config["FeatureDim"], 
                                   num_hidden_layers   = config["NumberOfBlocks"],
                                   num_attention_heads = config["NumberOfHeads"],
                                   intermediate_size   = config["FFdims"],
                                   hidden_act          = config["Activation"],
                                   hidden_dropout_prob          = 0.1,
                                   attention_probs_dropout_prob = 0.1,
                                   max_position_embeddings      = 512,
                                   type_vocab_size              = 2,
                                   initializer_range            = 0.02,
                                   layer_norm_eps               = 1e-12,
                                   pad_token_id                 = 0,
                                   position_embedding_type = 'absolute',
                                   use_cache   = True,
                                   return_dict = True,
                                   training    = True)
    model = HuggingFaceModel(Config)
    return model
    
