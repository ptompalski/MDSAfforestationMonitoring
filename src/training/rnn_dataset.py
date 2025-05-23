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
    2. Shuffles the lookup table within 'Age' groups to enable randomised batches while minimising sequence padding.
    3. For each row in the lookup table, loads the corresponding variable-length satellite sequence.
    4. Returns a tuple containing site features, satellite sequence, and target value as torch tensors.

    Parameters
    ----------
    lookup_dir : str
        Path to the Parquet file containing the site features, target values, and file references.
    seq_dir : str
        Directory containing the satellite data files. The 'file_name' column in the lookup table specifies the file to load for each row.
    site_cols : List of str
        List of columns in the lookup table to be used as site features.
    seq_cols : List of str
        List of columns in the sequence to be used as satellite features. `len(seq_cols)` should match the `input_size` for the rNN model.

    Attributes
    ----------
    original_lookup : pd.DataFrame
        The full, unshuffled lookup DataFrame loading from the lookup parquet file.
    lookup : pd.DataFrame
        A shuffled copy of the lookup table, regenerated when `reshuffled()` is called.
        Samples are randomly shuffled within 'Age' groups to optimise batch efficiency.
    
    Methods
    -------
    reshuffle() : 
        Regenerates a randomised lookup table by shuffling samples within 'Age' groups. Should be called at the start of each training epoch.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - site_features : torch.Tensor of shape [num_site_features,]
            Site records.
        - sequence : torch.Tensor of shape [seq_len, input_size]
            Variable-length satellite data.
        - target : torch.Tensor of shape [1,]
            Survival rate.

    Notes
    ------
    A zero-filled tensor of shape (1, 10) is returned if the sequence file fails to load.
    """
    def __init__(
            self, 
            lookup_dir : str, 
            seq_dir : str, 
            site_cols : List[str],
            seq_cols : List[str]
        ):
                
        self.original_lookup = pd.read_parquet(lookup_dir)
        self.seq_dir = seq_dir
        self.site_cols = site_cols
        self.seq_cols = seq_cols
        self.reshuffle()
        
    def __len__(self):
        """
        Return the number of samples in the dataset, that is the number of survival records in the lookup table.
        """
        return len(self.lookup)
    
    def reshuffle(self):
        """
        Regenerates a randomised lookup table by shuffling samples within 'Age' groups.
        """
        self.lookup = self.original_lookup.groupby(
            'Age').sample(frac=1, replace=False).reset_index(drop=True)

    def __getitem__(self, idx : int):
        """
        Load the satellite data and return a dictionary containing the site data, satellite data and target as Pytorch tensors.
        """
        row = self.lookup.iloc[idx]
        seq_path = os.path.join(self.seq_dir, row['file_name'])
        site_features = torch.tensor(row[self.site_cols].values, dtype=torch.float32)
        target = torch.tensor(row['target'], dtype=torch.float32)
        try:
            sequence = torch.tensor(pd.read_parquet(seq_path, columns=self.seq_cols), dtype=torch.float32)
        except FileNotFoundError:
            sequence = torch.zeros((1, len(self.seq_cols)), dtype=torch.float32)
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
            - site_features : torch.Tensor of shape [num_site_features,]
                Site records.
            - sequence : torch.Tensor of shape [seq_len, input_size]
                Variable-length satellite data.
            - target : torch.Tensor of shape [1,]
                Survival rate.

    Returns
    --------
    Dict[str, torch.Tensor]
        - site_features : torch.Tensor of shape [batch_size, ]
            Stacked site records.
        - sequence : torch.Tensor of shape [batch_size, max_seq_len, input_size]
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


def dataloader_wrapper(
    lookup_dir : str,
    seq_dir : str,
    batch_size : int = 32,
    num_workers : int = 0,
    pin_memory : bool = True,
    site_cols: List[str] = ['Density', 'Type_Conifer',
                            'Type_Decidous', 'Type_Mixed', 'Age'],
    seq_cols: List[str] = ['DOY', 'neg_cos_DOY', 'log_dt', 'NDVI', 'SAVI',
                           'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR',
                           'TCB', 'TCG', 'TCW']
):
    """
    Creates a PyTorch Dataset and DataLoader for batching satellite sequence data with site features.

    This function:
    1. Initializes a custom Dataset that loads site records and their corresponding satellite records.
    2. Initiates a DataLoader using a custom collate function to handle variable-length sequences.
 
    Parameters
    ----------
    lookup_dir : str
        Path to the Parquet file containing the lookup table.
    seq_dir : str
        Directory containing the satellite data files referenced in the lookup table.
    batch_size : int, default=32
        Number of samples per batch.
    num_worker : int, default=0
        Number of subprocesses to use for data loading.
    pin_memory : bool, default=True
        If True, data is copied into device/CUDA pinned memory before returning. 
    site_cols : List of str, default=['Density', 'Type_Conifer', 'Type_Decidous', 'Type_Mixed', 'Age']
        List of columns in the lookup table to be used as site features.
    seq_cols : List of str, default=['DOY', 'neg_cos_DOY', 'log_dt', 'NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG', 'TCW']
        List of columns in the sequence to be used as satellite features. `len(seq_cols)` should match the `input_size` for the rNN model.

    Returns
    -------
    Tuple[Dataset, DataLoader]
        - dataset : Dataset
            Instantiated custom Dataset for site and satellite data.
        - loader : DataLoader
            PyTorch DataLoader with batching and custom collation for variable-length sequences.
    
    Raises
    ------
    ValueError
        If any of the following conditions occur:
        - `lookup_dir` or `seq_dir` is not a string.
        - `site_cols` or `seq_cols` is not a list of strings.

    Notes
    -----
    - Shuffling is disabled to preserve sample order for efficient dynamic padding of variable-length sequences.
    """
    # Exception handling
    for name, var, exp_type in [
        ("lookup_dir", lookup_dir, str),
        ("seq_dir", seq_dir, str),
        ("site_cols", site_cols, list),
        ("seq_cols", seq_cols, list),
        ]:
        if not isinstance(var, exp_type):
            raise ValueError(f'"{name}" expects {exp_type.__name__}, got {type(var).__name__}')
        if exp_type == list and not all(isinstance(col, str) for col in var):
            error_type = set(type(col).__name__ for col in var)
            raise ValueError(f"'{name}' expects a list of str, got {error_type}")
    
    dataset = AfforestationDataset(lookup_dir=lookup_dir, seq_dir=seq_dir, site_cols=site_cols, seq_cols=seq_cols)
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        collate_fn=collate_fn
    )
    return dataset, loader


