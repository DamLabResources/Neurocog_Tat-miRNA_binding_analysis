import pandas as pd
import numpy as np
from itertools import combinations
from tqdm.notebook import tqdm
import torch
from torch.utils.data import IterableDataset

class Uniprot2Species():
    """
    
    """
    
    def __init__(self, datapath = "../Data/RNAInter_w_seqs_current.csv"):
        df = pd.read_csv(datapath)
        self.uniprot2species = dict(zip(df['UniprotID'], df['Also_species']))
        
    def map_uniprot_id_to_species(self, uniprot_id):
        """
        links a single id to its domains. If it's not in the dict, returns None
        """
        return self.uniprot2species.get(uniprot_id)
    
    def map_uniprot_ids_to_species(self, uniprot_ids):
        """
        iterates map_uniprot_id_to_domains over many domains
        """
        
        return [self.map_uniprot_id_to_species(uniprot_id) for uniprot_id in uniprot_ids]
    
class Uniprot2GO():
    """
    TO DO:
        > Get GO Terms File to use for retriever
        > Load GOATOOLs object for Go Term assignment
        > Create functions to convert uniprot ID to GO terms
        > return GO terms as sets for jacard score calculation
        
    ---
    Container for converting uniprot IDs to their corresponding
    GO terms.
    

    """
    
    def __init__(self, uniprot_ids,
                 datafile = '/home/robertlink/Thesis_work/Chapter_2/RPITER/Data/swissprot_go_terms.tab'):
        
        # Load data and remove NaN entries
        df = pd.read_csv(datafile, sep = '\t')
        df = df.dropna()
        df = df[df['Entry'].isin(uniprot_ids)]
        
        # Convert long GO string into a set of all GO terms
        df['Gene ontology IDs'] = df['Gene ontology IDs'].str.split("; ").apply(set)
        
        # Convert the Dataframe into a dict converting UniProt IDs to their GO terms
        self.uniprot2go = pd.Series(df['Gene ontology IDs'].values, index = df['Entry']).dropna().to_dict()
    
    def map_uniprot_id_to_go_terms(self, uniprot_id):
        """
        links a single id to its domains. If it's not in the dict, returns None
        """
        return self.uniprot2go.get(uniprot_id)
    
    def map_uniprot_ids_to_go_terms(self, uniprot_ids):
        """
        iterates map_uniprot_id_to_domains over many domains
        """
        
        return [self.uniprot2go.get(uniprot_id) for uniprot_id in uniprot_ids]
    
class Uniprot2Domain():
    """
    It works!
    ---
    
    Container for converting protein names to their
    domain

    Relies on the output of PFam's pfam_scan.pl script on the entirety of
    SwissProt's database. Script was acquired from:
    ftp://ftp.ebi.ac.uk/pub/databases/Pfam/Tools/PfamScan.tar.gz
    
    Used default arguments (i.e. just fasta input and directory of PFam files)
    when converting sequences to their domains
    
    Creates a dictionary linking the uniprot ID to the set of domains found
    within their corresponding seuqences. Used as the basis for the class'
    two main functions:
    ---
    
         map_uniprot_id_to_domains: A wrapper to link one uniprot ID to its
                                    corresponding domains. If ID isn't in the
                                    list, it returns a None value
                                    
        map_uniprot_ids_to_domains: Applies map_uniprot_id_to_domains over many
                                    uniprot_ids
    """
    
    def __init__(self, uniprot_ids,
                 pfam_scan_output_path = "../Data/PfamScan/uniprot_sprot_domains.tsv"):
        
        #############################################################################
        def _get_uniprot_id_domains(domain_info : pd.DataFrame, uniprot_id : str):
            """
            Returns the set of all domains belonging to a specific uniprot_id using 
            inputted DataFrame from 
            """
            # Filter all domain info for a specific uniprot_id
            domains = domain_info[(domain_info["seq id"] == uniprot_id)]
            
            # Removes duplicate domains and converts to a set
            unique_domains   = set(domains["hmm acc"])
            
            return unique_domains
        
        def _create_uniprot_to_domain_dict(pfam_scan_output_path):
            """
            
            """
            # load pfam_scan output data
            domain_info = pd.read_csv(pfam_scan_output_path, delim_whitespace=True, skiprows = 28, 
                                      names = ["seq id","alignment start","alignment end","envelope start", 
                                               "envelope end","hmm acc","hmm name","type","hmm start", 
                                               "hmm end","hmm length","bit score","E-value","significance","clan"])            
            # Convert ID to UniprotID
            domain_info['seq id'] = domain_info['seq id'].apply(lambda x: x.split('|')[1])
            
            # Remove Non-present IDs
            domain_info = domain_info[domain_info['seq id'].isin(uniprot_ids)]
            
            # Turn off if you want both domains and families instead of domains
            domain_info = domain_info[(domain_info["type"] == 'Domain')]
            
            # Remove non-significant domains
            domain_info = domain_info[domain_info["significance"].map({1: True, 0: False})]
            
            # Create Uniprot --> Domain dict
            domain_info     = pd.Series(domain_info['hmm acc'].values, index = domain_info['seq id'])
            conversion_dict = domain_info.groupby(level=0).agg(set).to_dict()
            
            return conversion_dict
        ##########################################################################
        
        self.uniprot2domain = _create_uniprot_to_domain_dict(pfam_scan_output_path)
    
    def map_uniprot_id_to_domains(self, uniprot_id):
        """
        links a single id to its domains. If it's not in the dict, returns None
        """
        return self.uniprot2domain.get(uniprot_id)
    
class UniProt2SequenceSimilarityScore:
    """
    It works!
    ---
    
    TBD:
        > Clarify the NSW scoring method
        > Y
    ---
    Finds the similarity score between two uniprot IDs form the 
    """
    
    def __init__(self, uniprot_ids, sw_paired_scores_path):
        """
        
        """
        
        def _generate_sequence_similarity_dict(sw_paired_scores_path):

                df = pd.read_csv(sw_paired_scores_path, sep = '\t', 
                                 header = None, usecols = [0,1,2])
                
                df[[0,1]] = df[[0,1]].applymap(lambda x: x.split('|')[1])
                df[2] /= 100
                
                # Filter out irrelevant IDs
                df = df[ df[[0,1]].isin(uniprot_ids).all(axis=1) ]
                
                x = zip(df[0], df[1], df[2])
                y = {(a,b):c for a,b,c in x}

                return y
            
        self.scoring_matrix = _generate_sequence_similarity_dict(sw_paired_scores_path)
            
    def find_sequence_similarity_score(self, uniprot1, uniprot2):
        """
        
        """
        
        return max([self.scoring_matrix.get((uniprot1,uniprot2), 0), 
                    self.scoring_matrix.get((uniprot2,uniprot1), 0)])
    
    def calculate_sequence_similarity(self, uniprot1, uniprot2):
        """
        Calculates the sequence similarity between two proteins, which is just the
        average of the normalized sw score for each id in each position
        """
                
        sequence_similarirty = self.find_sequence_similarity_score(uniprot1, uniprot2)
        return sequence_similarirty
    
class FIREDataFrameGenerator():
    """
    TODO:
        > In "", shouldn't be returning 0 just because its missing information. Fix this
        > Instead of 0.5, drop values w/ no entry
    """
    
    def __init__(self, uniprot_ids, share_species = True):
        
        self.uniprot_ids   = uniprot_ids
        self.share_species = share_species
        
        # Conversion tools
        self.name2species          = Uniprot2Species("../Data/RNAInter_w_seqs_current.csv")
        self.name2go               = Uniprot2GO(self.uniprot_ids, '/home/robertlink/Thesis_work/Chapter_3/Data/swissprot_go_terms.tab')
        self.name2domains          = Uniprot2Domain(self.uniprot_ids, "../Data/PfamScan/uniprot_sprot_domains.tsv")
        self.names2seq_similarirty = UniProt2SequenceSimilarityScore(self.uniprot_ids, '/home/robertlink/Thesis_work/Chapter_3/Data/swissprot_all_v_all.tsv')
        
        def _generate_uniprot_info_dict():
            """
            Dictionary structure:
            Info_dict
               |__ [<uniprot_id>]
                      |__ [<protein species>]
                      |__ [<protein GO terms>]
                      |__ [<protein Pfam domains>]
            """
            
            info_dict = dict()
            
            for uniprot_id in self.uniprot_ids:
                info_dict[uniprot_id] = {"species"  : self.name2species.map_uniprot_id_to_species(uniprot_id),
                                         "domains"  : self.name2domains.map_uniprot_id_to_domains(uniprot_id),
                                         "go_terms" : self.name2go.map_uniprot_id_to_go_terms(uniprot_id)}
                
            return info_dict
        
        self.protein_info = _generate_uniprot_info_dict()
    
    def calculate_jaccard_score(self, set1, set2):
        """
        Returns jacord score of two sets (ratio between intersection and union)
        """
        
        return len(set1.intersection(set2)) / len(set1.union(set2))
    
    def calculate_similarity(self, uniprot1, uniprot2, key):
        """
        common function used to calculate both the functional simialrity
        and the domain similarity. 
        
        If a key does not have a 
        """        
        
        try:
            similarity = self.calculate_jaccard_score(self.protein_info[uniprot1][key], 
                                                      self.protein_info[uniprot2][key])
            return similarity
        
        except:
            return None
        
    
    def calculate_similarity_score(self, uniprot1, uniprot2):
        """
        
        """
        # Calculate all 3 measures to evaluation protein similarity
        seq_similarity_score = self.names2seq_similarirty.calculate_sequence_similarity(uniprot1, uniprot2)
        func_score           = self.calculate_similarity(uniprot1, uniprot2, "go_terms")       
        domain_score         = self.calculate_similarity(uniprot1, uniprot2, "domains")
        
        # Average these values for single score
        try:
            similarity_score = np.mean([seq_similarity_score, func_score, domain_score])
        except:
            similarity_score = None
        
        return seq_similarity_score, func_score, domain_score, similarity_score
    
    def filter_out_mixed_species_pairs(self, combinations):
        """
        
        """
        
        species = lambda uniprot_id: self.protein_info[uniprot_id]["species"]
        return [pair for pair in combinations if species(pair[0]) == species(pair[1]) ]
    
    def generate_similarity_df(self):
        """
        X
        """
        
        print("generating combinations...")
        uniprot_combonations = combinations(self.uniprot_ids, 2)
        print("")
        
        if self.share_species:
            print("only using shared species...")
            uniprot_combonations = self.filter_out_mixed_species_pairs(uniprot_combonations)
        
        print("calculating similarity scores...")
        
        uniprot_similarity_dict = {"seq_similarity" : dict(), "func_similarity" : dict(), "domain_similarity" : dict(),
                                   "similarity_score" : dict()}
        
        for pair in tqdm(uniprot_combonations):
            seq_similarity_score, func_score, domain_score, similarity_score = self.calculate_similarity_score(*pair)
            
            uniprot_similarity_dict["seq_similarity"][pair]    = seq_similarity_score
            uniprot_similarity_dict["func_similarity"][pair]   = func_score
            uniprot_similarity_dict["domain_similarity"][pair] = domain_score
            uniprot_similarity_dict["similarity_score"][pair]  = similarity_score
        
        # Creates df and sorts by ascending order. Drops None scores
        similarity_frame = pd.DataFrame(uniprot_similarity_dict).sort_values(by=['similarity_score'])
        similarity_frame = similarity_frame.dropna()
                
        # Unpacks the values in protein_pair as their own columns and deletes
        # the protein_pair column
        similarity_frame['uniprot1'], similarity_frame['uniprot2'] = zip(*similarity_frame.index)
        similarity_frame.index = range(len(similarity_frame))
        
        # Reorders cols for aesthetic purposes
        similarity_frame = similarity_frame[['uniprot1','uniprot2','seq_similarity','func_similarity','domain_similarity','similarity_score']]
        
        # If share_species, which species does each protein belong to?
        if self.share_species:
            similarity_frame['species'] = similarity_frame['uniprot1'].apply(lambda uniprot_id: self.protein_info[uniprot_id]["species"])
        
        return similarity_frame
    
class FIRE():
    """
    To do:
        > Incorperate Negative Samples argument. Defual
        
    ---
    Mathod acquired from: Cheng et al. 2017 (PMID: 28361676)
    
    Uses homology based shuffling to generate negative RPI samples. This method measures the similarity
    between two proteins and generates new RPIs by replacing the protein found in the original RPI with
    a protein with a low similarity score. The logic driving this being that proteins that are more 
    dissimilar to one another are less likely to interact with the same RNAs
    
    FIRE measures protein similarity using the average of three measures:
    
       Protein similarity: Performs a smith-waterman alignment and returns a normalized
                           version of the score (Fix me for later)
                           
    Functional similarity: Returns the ratio of GO terms shared by two proteins compared
                           to culmulation of all their GO terms
                           
        Domain similarity: Returns the ratio of PFam domains shared by two proteins comapred
                           to the cumulation of all their domains
    
    The main function used in FIRE is "generate_negative_samples", which inputs
    a paired list or zip object and measures similarites between all proteins in
    the list. 
    """
    
    def __init__(self, similarity_table,
                 protein_info_cols,
                 seed,
                 max_samples_per_pair = None):
        
        # FIRE output parameters
        self.similarity_table     = similarity_table
        self.prot_info_cols       = protein_info_cols
        self.seed                 = seed
        self.max_samples_per_pair = max_samples_per_pair
        
    def isolate_rna_prot_interactions(self, positive_df, uniprot_id):
        return positive_df[(positive_df['UniprotID'] == uniprot_id)]
    
    def isolate_protein_seq_and_info(self, uniprot_block):
        return uniprot_block[self.prot_info_cols].iloc[0].values
    
    def create_negative_sample_df(self, uniprot_block1, uniprot_block2):
        return pd.concat([uniprot_block1.iloc[:self.max_samples_per_pair,:], 
                          uniprot_block2.iloc[:self.max_samples_per_pair,:]], axis = 0)
        
    def create_negative_samples(self, uniprot_id1, uniprot_id2, positive_df):
        """
        X
        """
        
        uniprot1_block = self.isolate_rna_prot_interactions(positive_df, uniprot_id1)
        uniprot2_block = self.isolate_rna_prot_interactions(positive_df, uniprot_id2)
        
        # Isolate protein seqs
        uniprot_seq_and_info1 = self.isolate_protein_seq_and_info(uniprot1_block)
        uniprot_seq_and_info2 = self.isolate_protein_seq_and_info(uniprot2_block)

        # Swap protein info
        uniprot1_block.loc[:,self.prot_info_cols] = uniprot_seq_and_info2
        uniprot2_block.loc[:,self.prot_info_cols] = uniprot_seq_and_info1

        # Combine blocks
        negative_sample_frame = self.create_negative_sample_df(uniprot1_block, uniprot2_block)
        
        return negative_sample_frame
    
    def _auto_calculate_max_samples(self, positive_df):
        number_unique_uniprot_ids = len(set(pd.concat((self.similarity_table['uniprot1'], self.similarity_table['uniprot2']))))
        self.negative_samples     = len(positive_df) / number_unique_uniprot_ids
    
    def _trim_extra_negative_samples(self, negative_df, diff):
        
        extra      = negative_df.sample(diff, random_state = self.seed)
        trimmed_df = negative_df.drop(extra.index)
        
        return trimmed_df
    
    def generate_negative_samples(self, positive_df):
        """
        
        """
        
        negative_df = pd.DataFrame()
        samples = 0
        
        if not self.max_samples_per_pair:
            self._auto_calculate_max_samples(positive_df)
        
        for _, uniprot_id1, uniprot_id2 in self.similarity_table[['uniprot1','uniprot2']].itertuples():
            
            if samples >= len(positive_df):
                break
            
            negative_samples = self.create_negative_samples(uniprot_id1, uniprot_id2, positive_df)
            
            negative_df = pd.concat((negative_df, negative_samples), axis = 0)
            
            samples += len(negative_samples)
        
        
        negative_df.index = pd.RangeIndex(len(negative_df))
        
        ################################################################
        # If there is a greater number of negative samples than positive 
        # samples randomly trim any exra samples  
        ################################################################
        sample_diff = max([len(negative_df) - len(positive_df), 0])
        if sample_diff:
            negative_df = self._trim_extra_negative_samples(negative_df, sample_diff)
            negative_df.index = pd.RangeIndex(len(negative_df))
            
        negative_df['interacts'] = 0
            
        return negative_df
    
def magnitude(t : torch.Tensor):
    """
    Takes magnitude of row-wise vectors
    
    ex):
    ---
    >>> x = torch.tensor([[1,1,1],
                          [3,4,0]],
                          dtype = float)
    >>> magnitude(x)
    tensor([1.7321, 5.0000], dtype=torch.float64)
    """
    return (t**2).sum(axis=1).sqrt()

class IterableProteinEmbedding(IterableDataset):
    """
    RENAME
    """
    def __init__(self, proteins, tokenizer, model, chunksize, max_len, cuda = True, reduce = True):
        self.protein_combos  = proteins
        self.tokenizer       = tokenizer
        self.model           = model
        self.chunksize       = chunksize
        self.max_len         = max_len
        self.reduce          = reduce
        
        # If reduce is false, the reduce function does not change
        # the embedder's output. Otherwise, it uses a later defined
        # mean pooling to reduce the vector to 1D
        
        # self._reduce_funcs = [lambda x: x, self.mean_pooling]
        
        self.cuda = cuda
        
        if self.cuda:
            self.model.to('cuda');
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings     = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded  = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings       = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask             = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def __iter__(self):
        # swap below as a protein comboooooooo
        
        for i in tqdm(range(0,len(self.protein_combos),self.chunksize)):
            
            #print(list(self.protein_combos[i:i+self.chunksize]))
            
            tokenized_protein = self.tokenizer(list(self.protein_combos[i:i+self.chunksize]),
                                               padding = 'max_length',
                                               max_length = self.max_len + 2,
                                               return_tensors='pt')
            
            if self.cuda:
                tokenized_protein.to('cuda');
            
            # x = partial(self.mean_pooling, attention_mask = tokenized_protein['attention_mask'])
            
            if self.reduce:
                yield self.mean_pooling(self.model(**tokenized_protein), tokenized_protein['attention_mask']).detach()
            else:
                yield self.model(**tokenized_protein)[0].detach()
            # find way to incorperate .detach()