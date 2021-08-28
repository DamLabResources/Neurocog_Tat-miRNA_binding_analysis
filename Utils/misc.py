from Bio import SeqIO

def flatten(l):
    return [j for i in l for j in i]

def filter_sequences_by_len_from_fasta(file, max_len):
    with open(file) as handle:
        return [str(record.seq) for record in SeqIO.parse(handle, 'fasta') if len(record.seq) <= max_len]