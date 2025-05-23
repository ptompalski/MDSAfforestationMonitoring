import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
from typing import List, Tuple

class AfforestationDataset(Dataset):
    def __init__(self, lookup_dir : str, seq_dir : str):
        self.lookup = pd.read_parquet(
            lookup_dir).groupby(['age']).sample(frac=1)
        self.seq_dir = seq_dir
        self.site_cols = ['Density', 'Type_Conifer',
                          'Type_Decidous', 'Type_Mixed', 'Age']
        
    def __len__(self):
        return len(self.lookup)
    
    def __getitem__(self, idx : int):
        row = self.lookup.iloc[idx]
        seq_path = os.path.join(self.seq_dir, row['file_name'])
        site_features = torch.tensor(row[self.site_cols].values, dtype=torch.float32)
        target = torch.tensor(row['target'], dtype=torch.float32)
        try:
            sequence = torch.tensor(pd.read_parquet(seq_path), dtype=torch.float32)
        except:
            sequence = torch.zeros((1, 10), dtype=torch.float32)
        return site_features, sequence, target
         

def collate_fn(batch : List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    site_features, sequence, target = zip(*batch)
    sequence_length = [len(i) for i in sequence]
    sequence = pad_sequence(sequence, batch_first=True)
    return {
        'site_features' : torch.stack(site_features),
        'sequence' : sequence,
        'target' : torch.stack(target),
        'sequence_length' : torch.tensor(sequence_length, dtype=torch.int16)
    }
