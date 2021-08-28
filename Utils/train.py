import random
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re

import fastai
from fastai.callback.all import * 
from fastai.basics import Learner
from fastai.torch_core import trainable_params

from .model_utils import DenseSAE, ConvolutionalSAE
from .model_blocks import MainModel

import matplotlib.pyplot as plt
import seaborn as sbn

class MutualDataRemover:
    """
    class designed to find any mutual entries found in two
    different datasets. Mutual entries are defined by shared 
    ids -- most likely a shared RNA and protein ID or sequence. 
    User must specifify columns names to search.
    
    Once found, MutualDataRemover returns the training set without
    any entries found in the evaluation set
    
    Its primary use the remove_mutual_entries_from_training function.
    This inputs a training and evaluation dataset and returns a 
    training dataset pruned of any mutual entries. But this is also
    used to remove any mutually found protein pairs for negative
    sampling. 
    """
    
    def __init__(self, col_names : list or tuple):
        self.col_names = col_names
        
    def generate_pairs(self, data : pd.DataFrame) -> zip:
        return zip(*[data[col] for col in self.col_names])
    
    def find_parsed_idx(self, train_pairs : zip, eval_pair_set : set) -> list:
        return [n for n,pair in enumerate(train_pairs) if pair not in eval_pair_set]
    
    def remove_mutual_entries_from_training(self, training_data : pd.DataFrame, 
                                            evaluation_data : pd.DataFrame) -> pd.DataFrame:
        
        train_pairs    = list(self.generate_pairs(training_data))
        eval_pair_set  = set(self.generate_pairs(evaluation_data))
        
        parsed_train_idx = self.find_parsed_idx(train_pairs, eval_pair_set)
        
        parsed_training_data = train.iloc[parsed_train_idx, :].reset_index(drop=True)
        
        return parsed_training_data
    
def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def main_model_splitter(model : MainModel) -> list:
    """
    Splits the module in such a way where the RNA and Prot blocks
    have identical lrs. The dense and final blocks, however, are
    split by layer. 
    """
    
    rna_module_params   = trainable_params(model.blocks['rna'])
    prot_module_params  = trainable_params(model.blocks['prot'])
    
    dense_module_params = trainable_params(model.dense) + trainable_params(model.final)
    
    splitter_output = [rna_module_params+prot_module_params,
                       *dense_module_params]
    
    return splitter_output

def dense_sae_splitter(model : DenseSAE) -> list:  
    """
    Correctly splits learning rates across current training DenseSAE layers. Meant
    to be used as the "splitter" argument in fastai Learner
    
    NOTE: need to reinitialize learner with your updated training step to apply
          new layer splits. 
    """
    split_layers = [trainable_params(model.hidden_layers[model.train_step-1]), 
                    trainable_params(model.hidden_layers[-model.train_step])]
        
    return split_layers

class LayerFreezer:
    """
    Designed to be used with
    """
    
    def freeze_single_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False        
    
    def freeze_weights(self, learner, step):
        """
        
        """
        learner.model.eval()
        self.freeze_single_layer(learner.model.hidden_layers[step-1])
        self.freeze_single_layer(learner.model.hidden_layers[-step])
        learner.model.train()
    
    def freeze_all_layers(self, learner):
        for layer in learner.model.hidden_layers:
            self.freeze_single_layer(layer)

    def unfreeze_single_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def unfreeze_all_layers(self, learner):
        for layer in learner.model.hidden_layers:
            self.unfreeze_single_layer(layer)     
    

class SAETrainer:
    """
    A wrapper class to hold all functions related to training SAEs
    """
    
    def __init__(self, epochs, seed, plot = True):
        self.epochs = epochs
        self.seed   = seed
        self.plot   = plot
            
        self.freezer = LayerFreezer()
    
    def fit_one_cycle(self, learner, lr_min, lr_steep):
        if lr_min >= lr_steep:
            learner.fit_one_cycle(self.epochs, lr_max = slice(lr_steep, lr_min))
        else:
            learner.fit_one_cycle(self.epochs, lr_max = slice(lr_min, lr_steep))
    
    def train_sae_step(self, learner, step):
        # Change behavior of training 
        learner.model.train_step = step
            
        lr_min, lr_steep = learner.lr_find(show_plot = self.plot)

        # Fit the model using your specified learner and epochs
        self.fit_one_cycle(learner, lr_min, lr_steep)
        self.freezer.freeze_weights(learner, step)
        
    def plot_train_val_loss(self, learner, title, savefile = None):
        
        fig, ax = plt.subplots()
        learner.recorder.plot_loss()
        
        sbn.despine()
        plt.ylabel("MSE Loss")
        plt.xlabel("Batches")
        plt.title(title)
        
        if savefile:
            plt.savefig(savefile, dpi = 300)
        
        plt.show()
        
    def train_sae(self, loaders, sae, sae_splitter_function, learner_cbs = None, loss_plot_dir = None) -> Learner:
        
        set_seed(self.seed)
        
        # Trains each layer individually and then finishes with
        # a single training session
        STEPS = len(sae.hidden_layer_arg)+1
        train_steps = range(1, STEPS+1)
        
        for step in train_steps:
            
            sae.train_step = step
            
            # Reinitialize learner to reapply splitter function
            
            if step < STEPS:
                FNC = sae_splitter_function
            else:
                FNC = None
            
            learner = Learner(loaders, sae,
                              loss_func = nn.MSELoss(),
                              splitter  = sae_splitter_function,
                              cbs       = learner_cbs, # [EarlyStoppingCallback(patience = 10)]
                              model_dir = '.'
                              )
            
            self.train_sae_step(learner, step)
            
            if self.plot:
                self.plot_train_val_loss(learner,
                                         title = f"SAE Phase {step} Train-Val Loss")
            
            # If all hidden layers are frozen, unfreeze them all for the
            # final training cycle
            if step == STEPS - 1:
                self.freezer.unfreeze_all_layers(learner)
                
            # Reinitialize trained SAE to put into another laoder
            sae = learner.model

        learner.model.train_step = 0
        self.freezer.freeze_all_layers(learner)
        learner.model.eval()
        
        return learner

def conv_sae_splitter(model : ConvolutionalSAE) -> list:
    """
    Correctly splits learning rates across current training ConvSAE layers. Meant
    to be used as the "splitter" argument in fastai Learner
    
    NOTE: need to reinitialize learner with your updated training step to apply
          new layer splits. 
    """
    
    step = model.train_step
    
    if step == 1:
        input_params  = trainable_params(model.input_layer)
        output_params = trainable_params(model.final_layer)
        
    else:
        input_params  = trainable_params(model.conv_layers[step-2])
        output_params = trainable_params(model.deconv_layers[-step+1])
        
    split_layers = [input_params, output_params]
        
    return split_layers
    
class ConvSAETrainer(SAETrainer):
    """
    SAE trainer modified to train ConvSAEs. The main difference is
    that there are slight modifications to "freeze_weights" and 
    "train_sae" to reflect the properties of the ConvSAE. 
    """
    
    def __init__(self, epochs, seed):
        super().__init__(epochs, seed)
        
    def freeze_weights(self, learner, step):
        """
        
        """
        learner.model.eval()
        
        if step == 1:
            self.freeze_single_layer(learner.model.input_layer)
            self.freeze_single_layer(learner.model.input_maxpool)
            self.freeze_single_layer(learner.model.final_upsample)
            self.freeze_single_layer(learner.model.final_layer)
            
        else:
            self.freeze_single_layer(learner.model.conv_layers[step-2])
            self.freeze_single_layer(learner.model.pooling_layers[step-2])
            self.freeze_single_layer(learner.model.upsampling_layers[-step+1])
            self.freeze_single_layer(learner.model.deconv_layers[-step+1])
        
        learner.model.train()
        
    def train_sae(self, loaders, sae, sae_splitter_function, learner_cbs = None, loss_plot_dir = None) -> Learner:
        
        set_seed(self.seed)
        
        STEPS = len(sae.conv_channel_sizes)
        train_steps = range(1, STEPS+1)
        
        for step in train_steps:
            
            sae.train_step = step
            
            # Reinitialize learner to reapply splitter function
            learner = Learner(loaders, sae,
                              loss_func = nn.MSELoss(),
                              splitter  = sae_splitter_function,
                              cbs       = learner_cbs, # [EarlyStoppingCallback(patience = 10)]
                              model_dir = '.'
                              )
            
            self.train_sae_step(learner, step)
            self.plot_train_val_loss(learner,
                                     title = f"SAE Phase {step} Train-Val Loss")
            
            # Reinitialize trained SAE to put into another laoder
            sae = learner.model
            
        learner.model.train_step = 0
        learner.model.eval()
        
        return learner
        
def set_seed(seed):
    """
    Sets all seed values to a consistant value for CPU and GPU applications. 
    
    Code acquired from:
    https://docs.fast.ai/dev/test.html#getting-reproducible-results
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

def save_train_val_loss_graph(fold, learner):
    "Saves the train-val loss graph as a png. Only generates for"
    fig, ax = plt.subplots()
    learner.recorder.plot_loss()

    sbn.despine()
    plt.ylabel("Log loss")
    plt.xlabel("Batches")
    #plt.title(title)

    plt.savefig(f"dummy_losses_fold_{fold}.png", dpi = 300)

class BlockInputShapeReformatter:
    """
    Modifies the protein and rna block inputs so that they're
    compatable with each model. 
    """
      
    def reformat_one_dimmention_input(self, block, input_tensor):
        if block == 'Reccurent':
            return input_tensor.unsqueeze(1)
        if block  == 'StackedRes':
            return input_tensor.unsqueeze(-1)
        return input_tensor
    
    def reformat_two_dimmention_input(self, block, input_tensor):
        if block == 'Dense':
            return input_tensor.flatten(1)
        if block == 'ConvPool':
            # Reformat to batchsize x position x embedding
            if input_tensor.shape[1] > input_tensor.shape[-1]:
                return torch.swapaxes(input_tensor,-1,1)
        return input_tensor
    
    def reformat_input(self, block, X):
        if len(X.shape) == 2:
            return self.reformat_one_dimmention_input(block, X)
        if len(X.shape) > 2:
            return self.reformat_two_dimmention_input(block, X)
        
class MainModelDenseInputSizeCalculator:
    """
    
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
        
        if len(block_out2.shape) == 2:
            block_out2 = self.pad1d_to_d2(block_out2)
            
        block_out2 = flatten(block_out2)
        return torch.cat([block_out1,block_out2],1)
    
    def calcualte_input_shape(self, rna_blocks, prot_blocks):
        rna_out  = self.pass_through_blocks(rna_blocks, self.rna_x)
        prot_out = self.pass_through_blocks(prot_blocks, self.prot_x)
        combined_out = self.flatten_block_outputs(*rna_out, *prot_out)
        outshape = combined_out.shape[-1]
        return outshape
    
# class SAEMLModelTrainer():
#     """
#     A trainer class designed to train ML models that rely on
#     SAE embedded data. 
    
#     Its main function is "train", which incorperates a 
#     ML model that you want to train along with training
#     data
#     """
    
#     def __init__(self, dense_sae):
#         self.dense_sae = dense_sae
    
#     def _preprocess_tensors(self, tensor, dtype = "float64"):
#         return tensor.numpy().astype(dtype)
    
#     def train(self, model, X_train, y_train):
#         X_train = self.dense_sae(X_train)
#         y_train = self._preprocess_tensors(y_train, int)
        
#         model.fit(X_train, y_train)
    
# class SupplementaryHIVRefiner:
#     """
#     A wrapper to hold model and model data to refine using
#     supplementary training methods
#     """
    
#     def __init__(self, tat_tar_learner, seed = 255):
        
#         self.learner = tat_tar_learner
#         self.epochs  = epochs
#         self.seed    = seed
        
#     def refine(self):
#         """
        
#         """
        
#         self.learner.fit_one_cycle(epochs)

# class ModelEvaluator:
#     """
    
#     """
    
#     def __init__(self, folds = 5, batchsize = 32, seed = 255):
#         super().__init__(batchsize, seed)
        
#         self.folds     = folds
#         self.batchsize = batchsize
#         self.seed      = seed
        
#         self.k_fold_spliter    = StratifiedKFold(n_splits=self.folds, shuffle = True, random_state = SEED)
#         self.test_val_splitter = StratifiedKFold(n_splits=2, shuffle = True, random_state = SEED)
        
#         self.metrics          = list()
#         self.pr_values        = list()
#         self.roc_curve_values = list()
    
#     def _initialize_train_test_val_data(self, df, train_idx, test_idx):
#         train = df.iloc[train_index]
#         test, val = train_test_split(df.iloc[test_index], 
#                                          test_size = 0.5, 
#                                          stratify = df['interacts'].iloc[test_index], 
#                                          random_state = SEED)        
#         return train, test, val
    
#     def _initialize_datasets(self, train, test, val, rna2vec, prot2vec):
#         train = RPIDataset(train, rna2vec, prot2vec)
#         test  = RPIDataset(test, rna2vec, prot2vec)
#         val   = RPIDataset(val, rna2vec, prot2vec)        
#         return train, test, val
        
#     def _initialize_learner(self, train, test, val, rna2vec, prot2vec):
#         """
        
#         """
#         train, test, val = self.initialize_datasets(train, test, val, rna2vec, prot2vec)

#         loaders = DataLoaders.from_dsets(train, val,
#                                          bs=BATCHSIZE,
#                                          device = "cuda:0")
#         learner = Learner(loaders, model,
#                           loss_func = nn.BCELoss(),
#                           cbs       = [EarlyStoppingCallback(patience = 10)],
#                           model_dir = '.'
#                           )
        
#         return learner

#     def _train_model(self, learner):
#         lr_min, lr_steep = learner.lr_find()
#         learner.fit_one_cycle(1000, lr_max = slice(lr_steep, lr_min))

#     def evaluate_model(self, model: nn.Module, df : pd.DataFrame):
#         """
        
#         """
        
#         untrained_params = model.state_dict()
#         current_fold     = 1
        
#         for train_index, test_index in k_fold_spliter.split(df, df['interacts']):
#             print(f"Training Fold {current_fold}...")
            
#             # Reset base model parameters
#             model.load_state_dict(untrained_params)
            
#             # Split the data into train-test-validation splits
#             train, test, val = self._initialize_train_test_val_data(df, train_index, test_index)
#             learner          = self._initialize_learner(train, test, val, rna2vec, prot2vec)
            
#             # Train the model and save 
#             self._train_model(learner)
#             learner.model.eval();
#             save_train_val_loss_graph(current_fold, learner)
            
#             # Properly format test data
#             test_rna, test_prot, y_true = test[:]
#             y_true    = y_true.numpy()
#             test_rna  = test_rna.cuda()
#             test_prot = test_prot.cuda()
            
#             # Predict on unseen test dataset
#             y_pred    = learner.model(test_rna, test_prot)
#             y_pred    = y_pred.cpu().detach().numpy()

#             # Evaluate model on test set and calculate performance metrics:
#             # ROC Curve values, PR Curve values
#             # accuracy, recall, specificity, f1, and MCC
#             self.roc_curve_values.append(roc_curve(y_true, y_pred))
#             self.pr_values.append(precision_recall_curve(y_true, y_pred))

#             self.metrics.append([accuracy_score(y_true, y_pred.round()),
#                                  recall_score(y_true, y_pred.round()),
#                                  specificity_score(y_true, y_pred.round()),
#                                  f1_score(y_true, y_pred.round()),
#                                  matthews_corrcoef(y_true, y_pred.round())])

#             current_fold += 1
        
#         def save_fold_roc_values(false_positive_rate_pth : str, true_positive_rate_pth : str) -> None:
            
#             false_positive_values, true_positive_values, _ = zip(*self.roc_curve_values)
            
#             fps_df = pd.DataFrame(false_positive_values)
#             tps_df = pd.DataFrame(true_positive_values)
            
#             fps_df.to_csv(recall_path)
#             tps_df.to_csv(precision_pth)
            
#         def save_fold_pr_values(precision_pth : str, recall_path : str) -> None:
            
#             precision_values, recall_values, _ = zip(*self.pr_values)
            
#             precision_df  = pd.DataFrame(precision_values)
#             recall_df     = pd.DataFrame(recall_values)
            
#             precision_df.to_csv(precision_pth)
#             recall_df.to_csv(recall_path)
            
#         def save_fold_metrics(metric_path : str, roc_path : str, pr_path : str) -> None:
#             metrics_df = pd.DataFrame(self.metrics, columns = ['Accuracy','Sensitivity','Specificity','F1','MCC'])
#             metrics_df.to_csv(metric_path)
            
