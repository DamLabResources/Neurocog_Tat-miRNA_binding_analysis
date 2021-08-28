import math
import string
from functools import reduce
from itertools import product, chain

from gensim.models import Word2Vec

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F 

from transformers import AutoTokenizer, AutoModel, pipeline
import sentencepiece

from tqdm.notebook import tqdm

class OneHotEncoder(nn.Module):
    """
    Layer to One-hot encode sequences
    """
    
    def __init__(self, molecule):
        
        assert molecule in ['rna','protein'], f"Your OneHotEncoder molecule arg must either be 'protein' or 'rna'. molecule arg is: {molecule}"
        
        self.type = molecule
        
        def _create_protein_dict():
            aas = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
            zeros = np.eye(20)
            
            return {aa: zeros[n,:] for n,aa in enumerate(aas)}
        
        self.dict = {'rna': {'A': [1, 0, 0, 0],
                             'U': [0, 1, 0, 0],
                             'T': [0, 1, 0, 0],
                             'G': [0, 0, 1, 0],
                             'C': [0, 0, 0, 1],
                             'N': [0.25]*4},
                     'protein': _create_protein_dict()}
                     
    def forward(x):
        return torch.tensor([self.dict[self.type][letter] for letter in x])
    
class KmerFrequencyEmbedder:
    """
    
    """
    
    def __init__(self, k, seq_alphabet):        
        assert type(k) == int, f"k must be an int, Inputted type is: {type(k)}"
        assert k > 0, f"k must be greater than 0. k is: {k}"
        
        self.k = k
        self.seq_alphabet = seq_alphabet
    
    def create_kmer_library(self, alphabet, k):
        return [''.join(kmer) for kmer in product(alphabet, repeat = k)]
    
    def create_normalized_frequency_vector(self, seq, kmer_library):
        """
        
        """
        frequency_vector = {kmer: seq.count(kmer) for kmer in kmer_library}
        total_kmers = sum(frequency_vector.values())
        normalized_frequency_vector = {key: value / total_kmers for key,value in frequency_vector.items()}
        torch_vector = torch.tensor(list(normalized_frequency_vector.values()), dtype = torch.double)
    
        return torch_vector
    
    def create_normalized_frequency_vectors(self, seqs, seq_alphabet = None):
        """
        
        """
        if seq_alphabet:
            kmer_library      = self.create_kmer_library(seq_alphabet, self.k)
            frequency_vectors = torch.stack([self.create_normalized_frequency_vector(seq, kmer_library) for seq in seqs])
        else:
            kmer_library      = self.create_kmer_library(self.seq_alphabet, self.k)
            frequency_vectors = torch.stack([self.create_normalized_frequency_vector(seq, kmer_library) for seq in seqs])
    
class ImprovedConjointTriadEmbedder(KmerFrequencyEmbedder):
    """
    Improved CTF embedding used by RPITER
    """
    
    def __init__(self, k, alphabet):
        super().__init__(k, alphabet)
    
    def _flatten(self, nested_list):
        return list(chain(*nested_list))
    
    def create_kmer_specific_normalized_frequency_vector(self, seq, k, alphabet = None):
        if alphabet:
            kmer_library = self.create_kmer_library(alphabet, k)
        else:
            kmer_library = self.create_kmer_library(self.seq_alphabet, k)
                
        return self.create_normalized_frequency_vector(seq, kmer_library)
        
    def create_improved_conjoint_triads_from_seq(self, seq):
        
        nested_improved_triad = [self.create_kmer_specific_normalized_frequency_vector(seq, k) for k in range(1,self.k+1)]
        improved_triads       = torch.cat(nested_improved_triad)
        
        return improved_triads
        
    def create_improved_conjoint_triads_from_seqs(self, seqs):
        return torch.stack([self.create_improved_conjoint_triads_from_seq(seq) for seq in seqs])
        
class ImprovedConjointStructTriadEmbedder(ImprovedConjointTriadEmbedder):
    """
    Improved struct CTF embedding used by RPITER
    
    NOTE: RPITER treats '[' and ']' as identical structures, putting RNA based embedded at 370 total featrues (assuming 4-mer)
    """
    
    def __init__(self, k, seq_alphabet, struct_alphabet):
        super().__init__(k, seq_alphabet)
        
        self.struct_alphabet = struct_alphabet
        
    def create_improved_conjoint_struct_triads_from_seq_and_ss(self, seq, ss):
        nested_improved_triad    = [self.create_kmer_specific_normalized_frequency_vector(seq, k, self.seq_alphabet) for k in range(1,self.k+1)]
        nested_improved_ss_triad = [self.create_kmer_specific_normalized_frequency_vector(ss, k, self.struct_alphabet) for k in range(1,self.k+1)]
        
        combined_nested_triads = nested_improved_triad + nested_improved_ss_triad
        improved_combined_triads = torch.cat(combined_nested_triads)
                
        return improved_combined_triads
        
    def create_improved_conjoint_struct_triads_from_seqs_and_structs(self, seqs, structs):
        return torch.stack([self.create_improved_conjoint_struct_triads_from_seq_and_ss(seq, ss) for seq,ss in zip(seqs, structs)])
    
class AlphabetReducer:
    """
    Transforms a protin sequence into a seuqence comprised of a reduced alphbet.
    By default, this is a standard 7-letter reduced alphbet based on mutual biochemical
    properties. 
    
    This can be altered by inputted a dictionary that maps each AA into a
    """
    
    def __init__(self, alphabet_mapping = {'A': 'RL1', 'G': 'RL1', 'V': 'RL1',
                                           'I': 'RL2', 'L': 'RL2', 'F': 'RL2', 'P': 'RL2',
                                           'Y': 'RL3', 'M': 'RL3', 'T': 'RL3', 'S': 'RL3',
                                           'H': 'RL4', 'N': 'RL4', 'Q': 'RL4', 'W': 'RL4',
                                           'R': 'RL5', 'K': 'RL5',
                                           'D': 'RL6', 'E': 'RL6', 
                                           'C': 'RL7'}):
        
        assert type(alphabet_mapping) == dict, f"The mapping dictionary must be a Dict. Input is type: {type(alphabet_mapping)}"
        assert len(alphabet_mapping) == 20, f"Your mapping dictionary maps {len(alphabet_mapping)} AAs. Must map 20 AAs."
        
        self.converter = alphabet_mapping
        
    def reduce_seq(self, protein_seq):
        return ''.join([self.converter[aa] for aa in protein_seq])
    
    def reduce_seqs(self, seqs):
        return [self.reduce_seq(seq) for seq in seqs]

class KmerSparseMatrixEmbedder(KmerFrequencyEmbedder):
    """
    
    """
    
    def __init__(self, k, seq_alphabet):
        super().__init__(k, seq_alphabet)
    
    def find_overlapping_kmers(self, seq):
        seq = seq.upper()
        return [seq[i:i+self.k] for i in range(len(seq)-self.k+1)]
    
    def generate_sparse_matrix_from_kmers(self, kmers):
        # Initialize temp kmer df
        df = pd.DataFrame(0, index = self.kmers, columns = range(len(kmers)))
        
        # Fill in df w/ kmer instances
        for n,kmer in enumerate(kmers):
            df.loc[kmer,n] = 1
        
        # Generate and return sparse matrix
        sparse_matrix = torch.tensor(df.values).float()
        return sparse_matrix
    
    def generate_sparse_matrix_from_seq(self, seq):
        kmers = self.find_overlapping_kmers(seq)
        sparse_matrix = self.generate_sparse_matrix_from_kmers(kmers)
        return sparse_matrix
    
class SequenceSVDEmbedder():
    """
    
    """
    
    def __init__(self, k, seq_alphabet):
        self.MatrixEmbedder = KmerSparseMatrixEmbedder(k, seq_alphabet)
        
    def embed_sparse_matrix(self, seq):
        return self.MatrixEmbedder.generate_sparse_matrix_from_seq(seq)
        
    def svd_embed_from_seq(self, seq):
        sparse_matrix = self.embed_sparse_matrix(seq)
        
        u,_,_ = torch.svd(sparse_matrix, some = False)
                
        # Mean of axis = 0 and axis = 1 is identical in all cases.
        # Using mean as it's more NN friendly
        vector = u.mean(axis=0)
        return vector
    
    def transform_seqs_to_svd_decomposition(self, seqs):
        sparse_matrices = [self.svd_embed_from_seq(seq) for seq in seqs]
        svd_decompositions = torch.stack(sparse_matrices)
        return svd_decompositions

class SkipGramEmbedder(nn.Module):
    """
    
    """
    
    def __init__(self, modelpath : str, reduce : bool, outsize = None):
        super().__init__()
        
        assert reduce or outsize, "SkipGramEmbedder cannot have reduce == False with no defined outsize"
        
        self.weightpath = modelpath
        self.bio2vec    = Word2Vec.load(self.weightpath)
        self.gram_size  = len(self.bio2vec.wv.index_to_key[0])
        self.reduce     = reduce
        self.outsize    = outsize
        
    def break_into_kemrs(self, seq):
        return [seq[i:i+self.gram_size] for i in range(len(seq)-self.gram_size+1)]
        
    def embed_seq(self, seq):
        kmers = self.break_into_kemrs(seq)
        kmer_matrix = torch.stack([torch.from_numpy(self.bio2vec.wv[kmer.lower()]) for kmer in kmers]).T
        
        if self.reduce:
            kmer_matrix = kmer_matrix.mean(dim=1)
        if not self.reduce and self.outsize:
            PADSIZE = self.outsize - len(seq) + 2
            kmer_matrix = F.pad(kmer_matrix, pad = (0, PADSIZE), mode='constant', value=0)
            
        return kmer_matrix
    
    def embed_seqs(self, seqs):
        return torch.stack([self.embed_seq(seq) for seq in seqs])
        
    def forward(self, seqs):
        return self.embed_seqs(seqs)

class ProtBERTEmbedder:
    """
    
    """
    
    def __init__(self, max_length, cuda):
        self.tokenizer  = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case = False)
        self.model      = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        
        self.max_length = max_length
        self.cuda       = cuda
        
        if self.cuda:
            self.model.to('cuda');
            
    def _format_protein(self, protein):
        return ' '.join(list(protein))
    
    def _format_proteins(self, proteins):
        return [self._format_protein(protein) for protein in proteins]
    
    def tokenize_proteins(self, formatted_proteins):
        tokenized_proteins = self.tokenizer(formatted_proteins,
                                            padding = 'max_length',
                                            max_length = self.max_length,
                                            return_tensors='pt')
        
        if self.cuda: 
            tokenized_proteins.to('cuda');
            
        return tokenized_proteins
    
    def generate_unpooled_embeddings(self, tokenized_proteins):
        return self.model(**tokenized_proteins)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings     = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded  = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings       = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask             = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def embded_tokenized_sequences(self,tokenized_proteins):
        with torch.no_grad():
            unpooled_embeddings = self.generate_unpooled_embeddings(tokenized_proteins)
            pooled_embeddings   = self.mean_pooling(unpooled_embeddings, tokenized_proteins['attention_mask']).detach()
            return pooled_embeddings
    
    def embed_sequences(self, sequences):
        formatted_proteins  = self._format_proteins(sequences)
        tokenized_proteins  = self.tokenize_proteins(formatted_proteins)
        pooled_embeddings = self.embded_tokenized_sequences(tokenized_proteins)
        return pooled_embeddings
    
    def _yield_embedded_sequences(self, sequences, chunksize):
        formatted_proteins  = self._format_proteins(sequences)
        
        for i in tqdm(range(0,len(formatted_proteins), chunksize)):
            tokenized_proteins  = self.tokenize_proteins(formatted_proteins[i:i+chunksize])
            pooled_embeddings = self.embded_tokenized_sequences(tokenized_proteins)
            yield pooled_embeddings
            
    def embed_sequences_low_memory(self, sequences, chunksize):
        return torch.cat([encoded_proteins for encoded_proteins in self._yield_embedded_sequences(sequences, chunksize)])