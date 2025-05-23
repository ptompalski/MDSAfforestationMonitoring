import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
from typing import List, Tuple

class AfforestationDataset(Dataset):
    """
    A custom PyTorch Dataset for loading site records and their corresponding satellite time series records.

    This dataset:
    1. Loads a lookup table containing site records, target values, and the file names of associated satellite data stored as Parquet files.
    2. Shuffles the lookup table within 'Age' groups to improve batching efficiency.
    3. For each row in the lookup table, loads the corresponding variable-length satellite sequence.
    4. Returns a tuple containing site features, satellite sequence, and target value as torch tensors.

    Parameters
    ----------
    lookup_dir : str
        Path to the Parquet file containing the site features, target values, and file references.
    seq_dir : str
        Directory containing the satellite data files. The 'file_name' column in the lookup table specifies the file to load for each row.

    Attributes
    ----------
    site_cols : List of str
        List of column names in the lookup table to be used as site features.
    lookup : pd.DataFrame
        Shuffled lookup DataFrame grouped by 'Age' to reduce padding in batching.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - site_features : torch.Tensor of shape [5,]
            Site records.
        - sequence : torch.Tensor of shape [seq_len, 10]
            Variable-length satellite data.
        - target : torch.Tensor of shape [1,]
            Survival rate.

    Notes
    ------
    - A zero-filled tensor of shape (1, 10) is returned if the sequence file fails to load.
    """
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
    """
    Collate function for batching variable-length satellite sequences with their corresponding site features and targets.

    This function:
    1. Sorts the batch in descending order by sequence length.
    2. Calculate the length of each sequence.
    3. Pads the sequences to the maximum seuquence length in the batch.
    4. Stacks site features and target tensors.
    5. Returns a dictionary with keys: `site_features`, `sequence`, `target` and `sequence_length`.

    Parameters
    ----------
    batch : List of Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A list of tuples where each tuple corresponds to a sample in the batch, containing:
            - site_features : torch.Tensor of shape [5,]
                Site records.
            - sequence : torch.Tensor of shape [seq_len, 10]
                Variable-length satellite data.
            - target : torch.Tensor of shape [1,]
                Survival rate.

    Returns
    --------
    Dict[str, torch.Tensor]
        - site_features : torch.Tensor of shape [batch_size, 5]
            Stacked site records.
        - sequence : torch.Tensor of shape [batch_size, max_seq_len, 10]
            Padded satellite sequences.
        - target : torch.Tensor of shape [batch_size,]
            Stacked target values.
        - sequence_length : torch.Tensor of shape [batch_size,]
            Sequence lengths before padding.
    """
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


def data_loader(
    lookup_dir : str,
    seq_dir : str,
    batch_size : int = 32,
    num_workers : int = 0,
    pin_memory : bool = True
):
    dataset = AfforestationDataset(lookup_dir=lookup_dir, seq_dir=seq_dir)
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        collate_fn=collate_fn
    )
    return dataset, loader


