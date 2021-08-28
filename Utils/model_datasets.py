from torch.utils.data import Dataset
import torch

class EncodedRPIDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.encoded_proteins = data[0]
        self.encoded_rnas = data[1]
        self.interaction = data[2].reshape(-1,1)
    
    def __len__(self):
        return len(self.encoded_proteins)
    
    def __getitem__(self, idx):
        return self.encoded_proteins[idx], self.encoded_rnas[idx], self.interaction[idx]


class SAEDataset(Dataset):
    """
    
    """
    
    def __init__(self, X):
        
        self.x = self.y = X.float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx,:]
    
class RPIDataset(Dataset):
    """
    
    """
    
    def __init__(self, df, rna2vec, prot2vec):
        super().__init__()
        
        self.rna2vec  = rna2vec
        self.prot2vec = prot2vec
        
        self.rna       = self.rna2vec(df['rnas'].str.lower().str.replace('t','u').values).float()
        self.prot      = self.prot2vec(df['proteins'].str.lower().values).float()
        self.interacts = torch.tensor(df['interacts'].values).unsqueeze(1).float()
        
    def __len__(self):
        return len(self.rna)
    
    def __getitem__(self, idx):
        return self.rna[idx], self.prot[idx], self.interacts[idx]
    
class HIVSupplementalDataset(Dataset):
    """
    ASSUMING PAIRS HAVE ALREADY BEEN MIXED IN THE BELOW FORM:
    Col 0: Original tat
    Col 1: Original TAR
    Col 2: Mixed Tat (either Col 0 or shuffled if Col 1)
    Col 3: Mixed TAR (either Col 1 or shuffled if Col 2)
    """
    
    def __init__(self, hiv_tat_tar_pair_df):
        
        self.x    = torch.tensor(hiv_tat_tar_pair_df.values)
        self.y    = -1
        
    def __len__(self):
        return len(hiv_tat_tar_pair_df)
    
    def __getitem__(self, idx):
        return self.x.iloc[idx, :], self.y