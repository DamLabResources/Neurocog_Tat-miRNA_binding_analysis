import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from fastai.layers import LinBnDrop, ConvLayer, MaxPool, AvgPool

from itertools import product
from gensim.models.callbacks import CallbackAny2Vec

import yaml
import pandas as pd

class HyperparameterParser:
    """
    When training the main model, hyperparameter opt
    might not have been performed for the specific
    RNA and protein block combos. Outputs of the
    hyperparameter opt experiments are in a csv 
    located in nni/experiment_results dir. If this 
    doesn't exist, then a set of default hyperparameters
    are used. These are defined in:
    Data/ModelHyperparameters/DefaultHyperparameters.yaml
    
    HyperparameterParser checks to see whether the csv exists
    and if not, uses the parameters defined in the default
    hyperparameters. 
    
    Regardless of file, HyperparameterParser loads and 
    parses it for input into the RNA or protein blocks through
    the "parse" function. 
    """
    
    def parse_yaml(self, file):
        with open(file) as handle:
            loaded_file = yaml.load(handle, Laoder=yaml.FullLoader)
            return loaded_file
            
    def __init__(self, default_params_file : str or None, experiemnt_dir : str or None):
        self.default        = default_params_file
        self.experiment_dir = experiemnt_dir
        
        self.block_specific_param_names = self.parse_yaml("Data/ModelHyperparameters/BlockSpecificHyperparametersName.yaml")
        
    def _split_model_name_into_blocks(self, model_name : str) -> tuple:
        _, rna_block, _, prot_block = model_name.split('_')
        return (rna_block, prot_block)
        
    def load_hyperparameters(self, model_name : str):
        try:
            optimized_parameter_csv = pd.read_csv(f"{self.experiment_dir}/{model_name}.csv")
            return optimized_parameter_csv
        
        except FileNotFoundError:
            default_hyperparameter_yaml = self.parse_yaml(self.default)
            return default_hyperparameter_yaml
        
    def parse_optimized_hyperparameters(self, csv : pd.DataFrame) -> (dict,dict):
        pass
    
    def parse_default_hyperparameters(self, yaml : dict) -> (dict,dict):
        pass
    
    def parse_hyperparameters(self, model_name : str) -> (dict,dict):
        model_params_file = self.load_hyperparameters(model_name)
        
        

class InputSizeCalculator:
    """
    Constructing the MainModel requires a concatination of the
    RNA and protein block outputs. The post-concatination size
    is variable depending on block hyperparameters and block types,
    making it impossible to calcualte the input size of the final
    dense layers a priori. 
    
    InputSizeCalculator serves to find that size by passing a
    dummy input into the RNA and protein blocks, concatinating
    that output, and returning the size needed to create the
    final part of MainModel. 
    
    This process is performed using the "calcualte_input_shape"
    function, which outputs the required integer. 
    """
    
    def __init__(self, dummy_rna_input, dummy_prot_input):
        self.rna_x  = dummy_rna_input
        self.prot_x = dummy_prot_input
    
    def pass_through_blocks(self, blocks, x):
        return [block(x) for block in blocks]
    
    def pad1d_to_d2(self, out):
        return out.unsqueeze(2)
    
    def flatten_block_outputs(self, block_out1, block_out2):
        flatten = nn.Flatten()
        
        if len(block_out1.shape) == 2:
            block_out1 = self.pad1d_to_d2(block_out1)
            
        block_out1 = flatten(block_out1)
        print('---')
        print(block_out1.shape)
        
        if len(block_out2.shape) == 2:
            block_out2 = self.pad1d_to_d2(block_out2)
            
        block_out2 = flatten(block_out2)
        print(block_out2.shape)
        print('---')
        return torch.cat([block_out1,block_out2],1)
    
    def calcualte_input_shape(self, rna_blocks, prot_blocks):
        
        rna_out  = self.pass_through_blocks(rna_blocks, self.rna_x)
        print(rna_out[0].shape)
        
        
        prot_out = self.pass_through_blocks(prot_blocks, self.prot_x)
#         if prot_out[0].shape[0] == 2:
#             prot_out[0] = prot_out[0][0].unsqueeze(0).unsqueeze(2)
    
        print(prot_out[0].shape)
        
        combined_out = self.flatten_block_outputs(*rna_out, *prot_out)
        print(combined_out.shape)
        
        outshape = combined_out.shape[-1]
        return outshape

class LossLogger(CallbackAny2Vec):
    """
    Callback to print and store train/val loss. Val loss isn't
    calculated by default, so let's fix that. 
    """
    # val_dataset, loss_function
    def __init__(self):
        self.current_epoch = 0
        
        #self.val_set  = val
        #self.loss     = val_loss
        
        self.epochs     = list()
        self.train_loss = list()
        self.val_loss   = list()
        
    def on_epoch_end(self, model):
        
        loss = model.get_latest_training_loss()
        
        if self.current_epoch == 0:
            print(f'Loss after epoch {self.current_epoch}: {loss}')
            # print(self.loss(model(self.val_set), self.val_set))
            self.train_loss.append(loss)
        else:
            mod_loss = loss - self.loss_previous_step
            print(f'Loss after epoch {self.current_epoch}: {mod_loss}')
            self.train_loss.append(mod_loss)
            
        self.loss_previous_step = loss
        self.epochs.append(self.current_epoch)
        self.current_epoch += 1

class _ActivationDropoutBase(nn.Module):
    """
    A utility class to prevent myself from defining the activation
    and dropout rates for everryyyyy subsequent model class 
    """
    
    def __init__(self, dropout, activation):
        super().__init__()
        
        self.activation = activation
        self.dropout    = dropout

class _AutoEncoderBase(_ActivationDropoutBase):
    """
    
    """
    
    def __init__(self, activation, batch_norm, dropout, train, lin_first):
        super().__init__(dropout, activation)
        
        self.batch_norm  = batch_norm
        self.train_model = train
        self.lin_first   = lin_first

class DenseAutoEncoder(_AutoEncoderBase):
    """
    ...
    """
    
    def __init__(self, input_output_shape,
                 hidden_layers = [256, 128, 64],
                 activation = nn.LeakyReLU(), 
                 batch_norm = True,
                 dropout = 0.2,
                 train = False,
                 lin_first = False):
        super().__init__(activation, batch_norm, dropout, train, lin_first)
        
        assert len(hidden_layers) > 0, "The hidden layer list is empty. It needs at least 1 hidden layer to be an AutoEncoder"
        
        # Calculates the list of all layer sizes in the AE
        self.shape = input_output_shape
        
        self.hidden_layer_arg = hidden_layers
        
        self.layer_sizes = [self.shape] + hidden_layers + hidden_layers[::-1][1:] + [self.shape]
        
        # Create and store all layers
        self.hidden_layers = nn.ModuleList([LinBnDrop(self.layer_sizes[i-1], self.layer_sizes[i], 
                                                      self.batch_norm,
                                                      self.dropout,
                                                      self.activation,
                                                      self.lin_first)
                              for i in range(1,len(self.layer_sizes)) ])
        
        # By design, traditional SAEs have an odd number of layers Because of this, the below will always return the middle
        # most layer in the SAE. Used for returning the stopping point of an untrained SAE
        self._latent_layer = int(len(self.hidden_layers) / 2)
        self.latent_size   = hidden_layers[-1]
        
    def forward(self, x):
        
        if self.train_model:
            for layer in self.hidden_layers:
                x = layer(x)
        else:
            for layer in self.hidden_layers[:self._latent_layer]:
                x = layer(x)
                
        return x

class DenseSAE(DenseAutoEncoder):
    """
    X
    """
    
    def __init__(self, input_output_shape : int,
                 hidden_layers = [256, 128, 64],
                 activation = nn.LeakyReLU(), 
                 batch_norm = True,
                 dropout    = 0.2,
                 lin_first  = False,
                 train_step = 0):
        super().__init__(input_output_shape, hidden_layers, activation, batch_norm, dropout)
        
        # Parameter used for training. This must be less than the 
        # length of the hidden_layers input. This designates which 
        # section of the SAE to train. If train_step is 0, then
        # the SAE processes the input until the latent layer
        self.train_step = train_step
                
    def forward(self, x):
        
        if 0 < self.train_step <= self._latent_layer:            
            layers = nn.ModuleList([*self.hidden_layers[:self.train_step], 
                                    *self.hidden_layers[-self.train_step:]])
            
        elif self.train_step > self._latent_layer:            
            layers = self.hidden_layers
            
        elif not self.train_step:
            layers = self.hidden_layers[:self._latent_layer]
            
        for layer in layers:
            # print(f"layer: {layer}")
            x = layer(x)
            
        return x

class ConvolutionalAutoEncoder(nn.Module):
    """
    TODO:
        > Completely rewrite so that it's big --> small --> big instead of
          size --> size --> size with temp layers. Remove temp layers!
        
    ---
    Just keep this architecture. There doesn't seem to be a consensus:
        1) Conv --> Deconv layers
        2) Conv --> Dense --> Deconv
        3) Conv+pooling --> Dense --> upsampling+deconv
        4) Conv+pooling --> upsampling+deconv
    """

    def __init__(self, input_output_channel_size,
                  input_seq_length,
                  conv_channel_sizes = [256, 128, 64],
                  pooling_sizes      = [2, 2, 2],
                  activation = nn.LeakyReLU):
        super().__init__()
            
        # Calculates the list of all layer sizes in the AE
        self.shape              = input_output_channel_size
        self.seq_len            = input_seq_length

        self.conv_channel_sizes = conv_channel_sizes
        
        if type(pooling_sizes) == int:
            self.pooling_sizes = [pooling_sizes]*len(self.conv_channel_sizes)
        else:
            self.pooling_sizes = pooling_sizes

        def generate_upsample_size_list(seq_len):
            upsample_size_list = list()

            for pool in self.pooling_sizes:
                upsample_size_list.append(int(seq_len))
                seq_len /=  pool

            # Needs to be reversed to use on otherside of conv ae
            upsample_size_list = upsample_size_list[::-1]

            return upsample_size_list

        self.upsample_sizes    = generate_upsample_size_list(self.seq_len)
        self.activation        = activation
 
        # Conv SAE Layers
        self.input_layer       = ConvLayer(self.shape, self.conv_channel_sizes[0], ndim = 1, act_cls = self.activation)
        self.input_maxpool     = MaxPool(self.pooling_sizes[0], ndim=1)
        
        self.conv_layers       = nn.ModuleList([ConvLayer(self.conv_channel_sizes[i-1], self.conv_channel_sizes[i], ndim = 1, act_cls = self.activation)
                                                 for i in range(1,len(self.conv_channel_sizes)) ])
 
        self.pooling_layers    = nn.ModuleList([MaxPool(kernel_size, ndim=1) for kernel_size in self.pooling_sizes[:-1]])
 
        self.upsampling_layers = nn.ModuleList([nn.Upsample(size = upsample_size) for upsample_size in self.upsample_sizes[:-1]])
 
        self.deconv_layers     = nn.ModuleList([ConvLayer(self.conv_channel_sizes[i], self.conv_channel_sizes[i-1], ndim = 1, padding=1, act_cls = self.activation, transpose=True)
                                                 for i in reversed(range(1,len(self.conv_channel_sizes))) ])
 
         # Final layer w/ no activation
        self.final_upsample    = nn.Upsample(size = self.upsample_sizes[-1])
        self.final_layer       = ConvLayer(self.conv_channel_sizes[0], self.shape, ndim = 1, padding=1, act_cls = None, norm_type=None, transpose=True)
        
    def forward(self, x):
        layer_count = len(self.conv_layers)
        #print(f"Initial input: {x.shape}")
        #print("---")
        
        x = self.input_layer(x)
        #print(f"Initial Conv out: {x.shape}")
        x = self.input_maxpool(x)
        #print(f"Initial Pool out: {x.shape}")
        
        for i in range(layer_count):
            x = self.conv_layers[i](x)
            #print(f"Conv{i} out: {x.shape}")
 
            x = self.pooling_layers[i](x)
            #print(f"Pool{i} out: {x.shape}")
 
        #print("---")
        for i in range(layer_count):
            x = self.upsampling_layers[i](x)
            #print(f"Upsample{i} out: {x.shape}")
 
            x = self.deconv_layers[i](x)
            #print(f"Deconv{i} out: {x.shape}")
        
        x = self.final_upsample(x)
        #print(f"Final upsample out: {x.shape}")
        x = self.final_layer(x)
        #print(f"Final deconv out: {x.shape}") 
        
        return x
    
class ConvolutionalSAE(ConvolutionalAutoEncoder):
    """
    Class using the improved CTF as input for the
    RNA and Protein sequence for both ImprovedCTF and ImprovedStructCTF
    """
    
    def __init__(self, input_output_channel_size,
                 input_seq_length,
                 conv_channel_sizes = [256, 128, 64],
                 pooling_sizes      = [2, 2, 2],
                 activation = nn.LeakyReLU, 
                 batch_norm = True,
                 train_step = 0):
        super().__init__(input_output_channel_size, input_seq_length, conv_channel_sizes, pooling_sizes, activation)
        
        # Parameter used for training. This must be less than the 
        # length of the hidden_layers input. This designates which 
        # section of the SAE to train. If train_step is 0, then
        # the SAE processes the input until the latent layer
        self.train_step = train_step
                
    def forward(self, x):
        
        x = self.input_layer(x)
        #print(f"Initial Conv out: {x.shape}")
        x = self.input_maxpool(x)
        #print(f"Initial Pool out: {x.shape}")
        
        STEPS = range(self.train_step-1)
        
        for i in STEPS:
            x = self.conv_layers[i](x)
            #print(f"Conv{i} out: {x.shape}")
 
            x = self.pooling_layers[i](x)
            #print(f"Pool{i} out: {x.shape}")
 
        #print("---")
        for i in reversed(STEPS):
            x = self.upsampling_layers[-i-1](x)
            #print(f"Upsample{-i-1} out: {x.shape}")
 
            x = self.deconv_layers[-i-1](x)
            #print(f"Deconv{-i-1} out: {x.shape}")
        
        x = self.final_upsample(x)
        #print(f"Final upsample out: {x.shape}")
        x = self.final_layer(x)
        #print(f"Final deconv out: {x.shape}") 
        
        return x