import tensorflow as tf
from HiConformer.model.RegressionLayer import Regression_Layer
from HiConformer.utils.Logger import init_logger,get_logger
from HiConformer.model.Activation_Selector import get_activation_function
from HiConformer.model.huggingface_loader import model_selector

def dimension_reduction_layer(input_shape: tuple, filters: int, kernels: list, poolsizes: list = None):
    H, W, C = input_shape
    activation = get_activation_function()
    input_tensor = tf.keras.Input(shape = input_shape)
    output_tensor = input_tensor
    for num_filter, kernel, poolsize in zip(filters, kernels, poolsizes):
        output_tensor = tf.keras.layers.Conv2D(filters = num_filter, kernel_size = kernel, padding = "same")(output_tensor)
        #print(f"first output tensor shape is :{output_tensor.shape}")
        #output_tensor = tf.keras.layers.LayerNormalization()(output_tensor)
        output_tensor = tf.keras.layers.BatchNormalization()(output_tensor) 
        output_tensor = tf.keras.layers.Activation(activation)(output_tensor)
        output_tensor = tf.keras.layers.MaxPool2D(pool_size = poolsize,
                                                 padding   = "same",
                                                 strides   = poolsize)(output_tensor)
        W /= poolsize[1]
    output_tensor = tf.keras.layers.AveragePooling2D(pool_size = (1, W), strides = (1, W), padding = "valid")(output_tensor)
    output_tensor = tf.keras.layers.Reshape((100, -1))(output_tensor)
    
    model = tf.keras.Model(input_tensor, output_tensor)
    return model

def HiConformer(seq1_config        : dict,
                seq2_config        : dict,
                avo1_config        : dict,
                avo2_config        : dict,
                window_config      : dict,
                transformer_config : dict,
                regressor_config   : dict,
                chromHMM_config    : dict,
                training_config    : dict ):

    activation = get_activation_function()

    # Sequence Input
    Seq1_input = tf.keras.layers.Input(shape=seq1_config['InputShape'], name = "Seq1_Input")
    Seq2_input = tf.keras.layers.Input(shape=seq2_config['InputShape'], name = "Seq2_Input")

    # Avocado
    Avo1_input = tf.keras.layers.Input(shape=avo1_config['InputShape'], name = "Avo1_Input")
    Avo2_input = tf.keras.layers.Input(shape=avo2_config['InputShape'], name = "Avo2_Input")

    Sequence_model = dimension_reduction_layer(seq1_config['InputShape'], 
                                               filters = seq1_config['NumFilters'], 
                                               kernels = [(1, kernel) for kernel in seq1_config['Kernels']],
                                               poolsizes = [(1, poolsize) for poolsize in seq1_config['PoolSizes']])
    Avocado_model = dimension_reduction_layer(avo1_config['InputShape'], 
                                              filters = avo1_config['NumFilters'], 
                                              kernels = [(1, kernel) for kernel in avo1_config['Kernels']], 
                                              poolsizes = [(1, poolsize) for poolsize in avo1_config['PoolSizes']])
    Seq1_output = Sequence_model(Seq1_input)
    Seq2_output = Sequence_model(Seq2_input)
    Avo1_output = Avocado_model(Avo1_input)
    Avo2_output = Avocado_model(Avo2_input)
                
    if training_config["SeqAvoMerge"]=="Concatenate":
        Bin1_output = tf.keras.layers.concatenate([Seq1_output, Avo1_output], axis=2) # (200,48) + (200,48) + (100,48)
    elif training_config["SeqAvoMerge"]=="Sum":
        assert avo1_config["NumFilters"][-1]==seq1_config["NumFilters"]
        Bin1_output = Seq1_output + Avo1_output
    else:
        raise NotImplementedError("Sequence embedding and Avocado embedding can only be merged by Concatenation or Summing!")

    embed_layer = tf.keras.layers.Dense(training_config["EmbedDim"])
      
    ChromHMM_Regressor = Regression_Layer(sizes = chromHMM_config['Bin1Nodes'],
                                          last_activation = "sigmoid",
                                          Dropout = 0.3,
                                          Output_dim = 13)
        
    if training_config["SeqAvoMerge"]=="Concatenate":
        Bin2_output = tf.keras.layers.concatenate([Seq2_output,Avo2_output],axis=2) # (200,48) + (200,48) + (100,48)
    elif training_config["SeqAvoMerge"]=="Sum":
        assert avo2_config["NumFilters"][-1]==seq2_config["NumFilters"]
        Bin2_output = Seq2_output + Avo2_output
    else:
        raise NotImplementedError("Sequence embedding and Avocado embedding can only be merged by Concatenation or Summing!")
            
    Bin1_output = embed_layer(Bin1_output)
    Bin2_output = embed_layer(Bin2_output)
    
    ChromHMM_bin1_output  = ChromHMM_Regressor(Bin1_output)
    ChromHMM_bin1_output  = tf.keras.layers.Lambda(lambda x: x, name = 'Bin1_ChromHMM_State')(ChromHMM_bin1_output)
    ChromHMM_bin2_output  = ChromHMM_Regressor(Bin2_output)
    ChromHMM_bin2_output  = tf.keras.layers.Lambda(lambda x: x, name = 'Bin2_ChromHMM_State')(ChromHMM_bin2_output)

    _,Bin1_L,_ = Bin1_output.shape
    _,Bin2_L,_ = Bin2_output.shape

    Transformer_input = tf.keras.layers.concatenate([Bin1_output,Bin2_output], axis=2) # (400,100)
    Transformer_input = tf.keras.layers.LayerNormalization()(Transformer_input)
    
    if transformer_config["UseHuggingFace"]:
        N                              = training_config["BatchSize"]
        transformer_config["EmbedDim"] = training_config["EmbedDim"] * 2
        model                          = model_selector(transformer_config)
        HuggingFace_input = {"inputs_embeds" : Transformer_input}
        Transformer_output = model(HuggingFace_input).last_hidden_state
    else:
        Transformer_embed = BertModelLayer(**BertModelLayer.Params(
                                            vocab_size               = -1,        # embedding params
                                            use_token_type           = True,
                                            use_position_embeddings  = True,
                                            token_type_vocab_size    = -1,
                                            num_layers               = transformer_config["NumberOfBlocks"],# transformer encoder params
                                            hidden_size              = training_config["EmbedDim"],
                                            num_heads                = transformer_config["NumberOfHeads"],
                                            hidden_dropout           = 0.4,
                                            intermediate_size        = 4 * training_config["EmbedDim"],
                                            intermediate_activation  = activation,
                                            adapter_size             = None,         # see arXiv:1902.00751 (adapter-BERT)
                                            shared_layer             = None,         # True for ALBERT (arXiv:1909.11942)
                                            embedding_size           = None,         # None for BERT, wordpiece embedding size for ALBERT
                                            ))(Transformer_input)
        
    Regressed_output = Regression_Layer(sizes           = regressor_config["RegressDims"],
                                        name            = "Interaction_Embed",
                                        Dropout         = regressor_config["DropoutRate"],
                                        Output_dim      = regressor_config["OutputDim"],
                                        last_activation = activation)(Transformer_output)
    
    Interaction_pred = Regression_Layer(sizes           = regressor_config["IFRegressDims"],
                                        name            = "Interaction_Frequency",
                                        Dropout         = regressor_config["IFDropoutRate"],
                                        Output_dim      = 1,
                                        last_activation = 'relu')(Regressed_output)

    # Create model
    model = tf.keras.models.Model(inputs  = [Avo1_input, Avo2_input, Seq1_input, Seq2_input], 
                                  outputs = [Interaction_pred,ChromHMM_bin1_output,ChromHMM_bin2_output],
                                  name    = 'HiConformer')
    return model




