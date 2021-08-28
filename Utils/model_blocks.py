import torch
from torch import nn
import fastai
from fastai.layers import LinBnDrop, SelfAttention, ResBlock, MaxPool, AvgPool, ConvLayer, Flatten

from functools import partial

class MainModel(nn.Module):
    """
    
    """
    
    def __init__(self, 
                 rna_blocks : list, prot_blocks : list, 
                 dense_input_size, dense_hidden_layers, 
                 **LinBnDrop_kwargs):
        super().__init__()
        
        self.blocks = nn.ModuleDict({"rna" : nn.ModuleList(rna_blocks),
                                     "prot": nn.ModuleList(prot_blocks)})
        self.flatten = nn.Flatten()
        self.dense = DenseBlock(dense_input_size, dense_hidden_layers, **LinBnDrop_kwargs)
        self.final = PredictionBlock(dense_hidden_layers[-1], **LinBnDrop_kwargs)
        
    def pass_through_blocks(self, x, molecule_type):
        return [block(x) for block in self.blocks[molecule_type]]

    def pad1d_to_d2(self, out):
        return out.unsqueeze(2)
    
    def _flatten_block(self, block):
        if len(block.shape) == 2:
            block = self.pad1d_to_d2(block)
        return self.flatten(block)
    
    def flatten_block_outputs(self, block_out1, block_out2):
        block_out1 = self._flatten_block(block_out1)
        block_out2 = self._flatten_block(block_out2)
        combined   = torch.cat([block_out1,block_out2],1)
        return combined
    
    def format_block_outputs(self, seq_outs):
        if seq_outs[0].shape == 2:
            seq_outs[0] = seq_outs[0][0].unsqueeze(0).unsqueeze(2)
        return seq_outs
    
    def forward(self, rna, prot):
        rna_outs = self.pass_through_blocks(rna, 'rna')
        rna_outs = self.format_block_outputs(rna_outs)
        
        prot_outs = self.pass_through_blocks(prot, 'prot')
        prot_outs = self.format_block_outputs(prot_outs)
        
        out = self.flatten_block_outputs(*rna_outs, *prot_outs)
        
        out = self.dense(out)
        out = self.final(out)
        
        return out

class MaxAvgPooling(nn.Module):
    """
    http://proceedings.mlr.press/v51/lee16a.pdf
    
    a ∈ [0,1]
    mix(x)   = a*max(x) + (1-a)*avg(x)
    
    below not used, but keep for later as it has better
    performance than mixed:
    gated(x) = sig(w.T*x)*max(x) + (1-sig(w.T*x))*avg(x)
    """
    
    def __init__(self, kernel_size, a, **pool_kwargs):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.a = a
        
        self.max = MaxPool(self.kernel_size, **pool_kwargs)
        self.avg = AvgPool(self.kernel_size, **pool_kwargs)
        
    def forward(self, x):
        
        max_out = self.max(x)
        avg_out = self.avg(x)
        
        mixed = self.a*max_out + (1-self.a)*avg_out
        
        return mixed

class SAEBlock(nn.Module):
    """
    Dense SAE:
    Batchsize x embedding vector
    
    Conv SAE:
    Batchsize x Sequencing position x embedding vector
    """
    
    def __init__(self, sae_type, sae_model_path):
        super().__init__()
        
        assert sae_type in ["Dense","Convolutional"], \
        f'SAEBlock uses either Dense or Convolutional as SAE type. Recieved: "{sae_type}" instead'
        
        self.sae_type = sae_type
        self.sae_model_path = sae_model_path
        
        self.sae = torch.load(self.sae_model_path)
        self.sae.eval()
        
    def forward(self, x):
        if self.sae_type == 'Dense':
            x = nn.Flatten()(x)
            
        x = self.sae(x)
        return x
        
class PredictionBlock(nn.Module):
    """
    Not meant to be used on its own. Only
    incorperated as tail end of MainModel
    """
    
    def __init__(self, input_channels, **LinBnDrop_kwargs):
        super().__init__()
        
        self.final   = LinBnDrop(input_channels, 1, **LinBnDrop_kwargs)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.final(x)
        x = self.sigmoid(x)
        return x

class DenseBlock(nn.Module):
    """
    
    """
    
    def __init__(self, input_size, hidden_layer_sizes, **LinBnDrop_kwargs):
        super().__init__()
        
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.input_layer = LinBnDrop(input_size, hidden_layer_sizes[0])
        
        self.hidden_layers = nn.ModuleList([LinBnDrop(hidden_layer_sizes[i-1], hidden_layer_sizes[i], **LinBnDrop_kwargs) 
                                            for i in range(1,len(self.hidden_layer_sizes))])
        
    def forward(self, x):
        
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            
        return x
    
class StackedResBlock(nn.Module):
    """
    need input side of
    
    1D:
    Batchsize x Embedding size
    * Input_size is the embedding size
    
    2d:
    Batchsize x sequence position x embedding size
    * Input_size is the seuqence position
    """
    
    def __init__(self, input_size, hidden_sizes, expansion = 1, **resblock_kwargs):
        super().__init__()
        
        self.hidden_sizes = hidden_sizes
        self.input_size   = input_size
        self.expansion    = expansion
        
        self.input_layer   = ResBlock(self.expansion, self.input_size, self.hidden_sizes[0], ndim=1, **resblock_kwargs)
        
        self.hidden_layers = nn.ModuleList([ResBlock(self.expansion, self.hidden_sizes[i-1], self.hidden_sizes[i], ndim=1,**resblock_kwargs)
                                            for i in range(1, len(self.hidden_sizes))])
        
    def forward(self, x):
        
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            
        return x
        
class AttentionBlock(nn.Module):
    
    def __init__(self, in_channels, layer_count):
        super().__init__()
        
        self.in_channels = in_channels
        self.layer_count = layer_count
        
        self.layers = nn.ModuleList([SelfAttention(self.in_channels) for _ in range(self.layer_count)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x

class ConvPoolBlock(nn.Module):
    "Try max pool, average pool, and average pool"
    
    def __init__(self, conv_channel_sizes, pooling_kernel_sizes, 
                 pooling_type, mixed_a = 0.5,
                 conv_kwargs = dict(), pool_kwargs = dict()):
        super().__init__()
        
        assert pooling_type in ['max','avg','mixed'] 
        assert len(conv_channel_sizes) >= 2
        
        if type(pooling_kernel_sizes) == int:
            self.pooling_kernel_sizes = [pooling_kernel_sizes for _ in range(len(conv_channel_sizes)-1)]
        
        self.conv_channel_sizes   = conv_channel_sizes
        
        assert 0 <= mixed_a <= 1, "mixed_a must be between 0 and 1"
        assert len(self.conv_channel_sizes)-1 == len(self.pooling_kernel_sizes), \
        "need equal number of conv_channel_sizes and pooling_kernel_sizes"
        
        self.pooling_type = pooling_type
        
        self.conv_kwargs = conv_kwargs
        self.pool_kwargs = pool_kwargs
        
        self.mixed_a = mixed_a
        
        self.pool = {'max'   : MaxPool,
                     'avg'   : AvgPool,
                     'mixed' : partial(MaxAvgPooling, a = self.mixed_a)}
        
        self.conv = nn.ModuleList([ConvLayer(self.conv_channel_sizes[i-1], self.conv_channel_sizes[i], ndim=1, **self.conv_kwargs) 
                                   for i in range(1, len(self.conv_channel_sizes))])
        
        self.pooling = nn.ModuleList([self.pool[self.pooling_type](kernel_size, ndim = 1, **self.pool_kwargs)
                                      for kernel_size in self.pooling_kernel_sizes])
        
    def forward(self, x):
        #print(f"INPUT SHAPE: {x.shape}")
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            #print(f"CONV OUTPUT SHAPE: {x.shape}")
            x = self.pooling[i](x)
            #print(f"POOL OUTPUT SHAPE: {x.shape}")
            
        return x

class ReccurentBlock(nn.Module):
    """
    Uses an LSTM or GRU
    
    """
    
    def __init__(self, input_size, reccurent_type, hidden_size, num_layers, ndim = 2, **recurrent_kwargs):
        super().__init__()
        
        assert reccurent_type.upper() in ['LSTM','GRU'], \
        f'ReccurentBlock uses either LSTM or GRU as reccurent type. Recieved: "{reccurent_type}" instead'
        
        self.type = reccurent_type.upper()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        
        self.reccruent_type_converter = {"LSTM": nn.LSTM, "GRU": nn.GRU}
        
        self.reccruent_block = self.reccruent_type_converter[self.type](self.input_size, self.hidden_size, self.num_layers, **recurrent_kwargs)
        
    def forward(self, x):
        
        out, _ = self.reccruent_block(x)
        return out