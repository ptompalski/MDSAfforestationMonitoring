import numpy as np
import pandas as pd
import os
import json
from src.data.pivot_data import pivot_df
import click
from pathlib import Path

def _get_summary_statistics(density_col: pd.Series, tc_cols: pd.DataFrame) -> dict:
    """
    Compute summary statistics (mean, std) for features requiring standard scaling.

    Separately computes:
    - Summary statistics for 'Density' from the lookup table
    - Summary statistics for 'TCW', 'TCG', 'TCB' from the remote sensing table

    Parameters
    ----------
    density_col : pd.Series
        The 'Density' column extracted from the lookup table.

    tc_cols : pd.DataFrame
        DataFrame containing the 'TCW', 'TCG', and 'TCB' columns from the remote sensing table.
        
    Returns
    -------
    dict
        Dictionary with keys 'mean' and 'std', each mapping to a dict of:
        - 'TCW', 'TCG', 'TCB', 'Density'
        Example:
        {
            'mean': {'TCW': ..., 'TCG': ..., ...},
            'std':  {'TCW': ..., 'TCG': ..., ...}
        }
    """
    stats_dict = {}
    stats_dict['mean'] = tc_cols.mean().to_dict()
    stats_dict['std'] = tc_cols.std().to_dict()
    
    stats_dict['mean']['Density'] = float(density_col.mean())
    stats_dict['std']['Density'] = float(density_col.std())
    
    return stats_dict


def _get_raw_sequence(
    site_key: dict, 
    remote_sensing_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Retrieve remote sensing records for a given site-pixel-date tuple.

    Filters the imaging dataset to all records for a specific (ID, PixelID)
    where ImgDate ≤ SrvvR_Date.

    Parameters
    ----------
    site_key : dict
        Unique identifiers of one survival record in the lookup table. Keys are:
        - ID
        - PixelID
        - SrvvR_Date

    remote_sensing_df : pd.DataFrame
        Full DataFrame of remote sensing records, must include:
        'ID', 'PixelID', 'ImgDate', 'DOY', and vegetation indices.

    Returns
    -------
    pd.DataFrame
        Sorted subset of imaging data for that record (ascending by ImgDate).
        Returns None if no matching remote sensing data.
    """    
    
    # apply filtering by ID, PixelID and image date
    filter_logic = (
    (remote_sensing_df['ID'] == site_key['ID']) &
    (remote_sensing_df['PixelID'] == site_key['PixelID']) &
    (remote_sensing_df['ImgDate'] <= site_key['SrvvR_Date']))
    
    raw_sequence = remote_sensing_df[filter_logic]
    
    # if no matching rows, return None 
    if raw_sequence.empty:
        return None
    else:
        return raw_sequence.sort_values(by='ImgDate')

def split_interim_dataframe(interim_df: pd.DataFrame) -> dict:
    """
    Split the interim feature-engineered DataFrame into:
    - A lookup table of survival records
    - A remote sensing table of vegetation index time series

    Parameters
    ----------
    interim_df : pd.DataFrame
        Partially cleaned DataFrame from preprocessing, containing both static
        and time-dependent features for each site-pixel-date combination.

    Returns
    -------
    dict
        {
            'lookup_df': pd.DataFrame of site features + target + SrvvR_Date,
            'remote_sensing_df': pd.DataFrame of vegetation indices with ImgDate and DOY
        }
    """
    remote_sensing_cols = [
    'ID','PixelID','ImgDate','DOY', # NOTE: FIX preprocess_features.py SO WE KEEP DOY
    'NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG', 'TCW' 
    ]
    
    lookup_cols = [
        'ID','PixelID','SrvvR_Date','Age','Density','Type','target'
    ]
    
    # get lookup table and remove duplicate rows if they exist (but they shouldn't), and convert to datetime
    lookup_df = pivot_df(interim_df).drop_duplicates()[lookup_cols]
    lookup_df['SrvvR_Date'] = pd.to_datetime(lookup_df['SrvvR_Date'])
    
    # some of the remote sensing cols are duplicated, should be dropped. Also convert ImgDate to datetime
    remote_sensing_df = interim_df[remote_sensing_cols].drop_duplicates()
    remote_sensing_df['ImgDate'] = pd.to_datetime(remote_sensing_df['ImgDate'])
    
    # type conversion for consistency
    lookup_df['target'] = lookup_df['target'].astype(float)
    lookup_df['Age'] = lookup_df['Age'].astype(int)
    
    return {
        'lookup_df':lookup_df,
        'remote_sensing_df': remote_sensing_df
    }
    

def process_and_save_sequences(
    lookup_df: pd.DataFrame,
    remote_sensing_df: pd.DataFrame,
    seq_out_dir: Path,
    lookup_out_path: Path,
    norm_stats: dict
) -> None:
    """
    Preprocess and save vegetation index time series for each (ID, PixelID, SrvvR_Date) triplet
    up to the survival rate record date. Each time series is saved as a separate Parquet file,
    and a lookup Parquet file is created for all site metadata and targets.

    Parameters
    ----------
    lookup_df : pd.DataFrame
        DataFrame of survival rate records, with columns:
        'ID', 'PixelID', 'SrvvR_Date', 'Age', 'Density', 'Type', 'target'.

    remote_sensing_df : pd.DataFrame
        Remote sensing DataFrame containing vegetation indices, with columns:
        'ID', 'PixelID', 'ImgDate', 'DOY', and vegetation features such as 
        'NDVI', 'NDWI', 'EVI', 'SAVI', 'MSAVI', 'TCW', 'TCG', 'TCB', 'NBR', etc.

    seq_out_dir : Path
        Directory where individual Parquet time series will be saved.

    lookup_out_path : Path
        Path where the consolidated lookup Parquet file will be written.

    norm_stats : dict
        Dictionary with mean and std values for 'TCW', 'TCG', 'TCB', and 'Density':
        {
            'mean': {'TCW': ..., 'TCG': ..., 'TCB': ..., 'Density': ...},
            'std':  {'TCW': ..., 'TCG': ..., 'TCB': ..., 'Density': ...}
        }

    Returns
    -------
    None
        Produces:
        - Parquet file per record: '<ID>_<PixelID>_<SrvvR_Date>.parquet'
        - Lookup table: e.g., 'lookup_train.parquet' with columns:
            * ID, PixelID, SrvvR_Date, Age, Density, target
            * One-hot columns: Type_Conifer, Type_Decidous, Type_Mixed

    Notes
    -----
    - Filters imaging data to only include records up to the survival record date.
    - Applies:
        * `log_dt = log(1 + Δt)` for time difference between survival and image date
        * `neg_cos_DOY = -cos(2π × DOY / 365)`
        * z-score normalization for TCW, TCG, TCB, and Density
    - Skips samples with no available imaging records, and drops them from the lookup table.
    """
    
    # get output directory for sequnce data
    seq_out_dir = Path(seq_out_dir)
    
    # standard scaling of the TCW, TCG, and TCB column
    for col in ['TCW', 'TCG', 'TCB']:
        mu, sigma = norm_stats['mean'][col], norm_stats['std'][col]
        remote_sensing_df[col] = (remote_sensing_df[col] - mu) / sigma
        
    # iterate through rows and get sequences and filenames for each.
    valid_indices = []
    fnames = []
    for idx, row in lookup_df.iterrows():
        
        # get key for site
        site_key = row[['ID','PixelID','SrvvR_Date']].to_dict()
        # for some reason type converts when I store in dict, need to convert back.
        site_key['SrvvR_Date'] = pd.to_datetime(site_key['SrvvR_Date'])
        
        # get raw sequence, do not add to lookup table if no sequence found
        # NOTE: in my EDA on training data it seems that ~12,500 records do not have matching sequences.
        sequence_df = _get_raw_sequence(site_key, remote_sensing_df)
        if sequence_df is None:
            continue
        
        # Feature engineering time columns
        time_delta = (site_key['SrvvR_Date'] - sequence_df['ImgDate']).dt.days
        sequence_df['log_dt'] = np.log1p(time_delta)
        sequence_df['neg_cos_DOY'] = -np.cos(2 * np.pi * sequence_df['DOY'] / 365)
        
        # drop DOY column
        sequence_df = sequence_df.drop(columns=['DOY'])  #NOTE: FIX preprocess_features.py SO WE KEEP DOY
        
        # save sequence as parquet file
        fname = f"{site_key['ID']}_{site_key['PixelID']}_{site_key['SrvvR_Date'].strftime('%Y-%m-%d')}.parquet"
        fnames.append(fname)
        sequence_df.to_parquet(seq_out_dir/fname, index=False)
        
        # store row
        valid_indices.append(idx)
        
    # filter lookup table for rows with sequences
    lookup_df = lookup_df.iloc[valid_indices]
    
    # Normalize Density
    lookup_df['Density'] = (
        lookup_df['Density'] - norm_stats['mean']['Density']
    ) / norm_stats['std']['Density']
    
    # one-hot encode Type variable, drop one redundant column to reduce dimension (might help with less model parameters later!)
    # Mixed can be treated as reference category.
    ohe_type = pd.get_dummies(lookup_df['Type'], prefix='Type',dtype=int).drop(columns=['Type_Mixed'])
    lookup_df = pd.concat([lookup_df.drop(columns='Type'), ohe_type], axis=1)
    
    # create filename column for lookup table and save
    lookup_df['filename'] = pd.Series(fnames)
    lookup_df.to_parquet(lookup_out_path, index=False)

@click.command()
@click.option(
    '--input_path', type=click.Path(exists=True,dir_okay=False), required=True,
    help='Path to partially cleaned Afforestation data, eg. data/interim/clean_feats_data.parquet'
)
@click.option(
    '--output_seq_dir', type=click.Path(exists=False,file_okay=False), required=True,
    help='Path to the directory in which sequence files will be stored, eg data/clean/sequences'
)
@click.option(
    '--norm_stats_path', type=click.Path(exists=False,dir_okay=False), required=False,
    default=os.path.join("data/interim/norm_stats.json"), show_default=True,
    help='''Path to the file that stores feature summary statistics for normaliztion.
            if no file exists at the path and --compute_norm_stats is True, 
            summary statistics will be computed and stored here.
         '''
)
@click.option(
    '--output_lookup_path', type = click.Path(dir_okay=False), required=True,
    help='Path to store lookup table, eg data/processed/train_lookup.parquet'
)
@click.option(
    '--compute-norm-stats/--no-compute-norm-stats',
    default=True,
    help='''Compute normalization statistics (default: True = compute from input data)
            Should not be computed on testing data.'''
)
def main(input_path,output_seq_dir,norm_stats_path,output_lookup_path,compute_norm_stats):
    '''
    Command-line interface for processing and saving time series for each survival rate record.
    '''
    # setup directories
    output_seq_dir = Path(output_seq_dir)
    output_seq_dir.mkdir(parents=True,exist_ok=True)
    norm_stats_path = Path(norm_stats_path)
    
    # Throw an error if stats_norm doesn't exist and user doesn't want to compute it
    if not norm_stats_path.is_file() and compute_norm_stats is False:
        raise FileNotFoundError(
             f"{norm_stats_path} does not exist.\n"
            f"Provide a correct path or rerun with --compute-norm-stats flag."
        )
    
    click.echo('Splitting into lookup and remote sensing dataframe...')
    # split interim data into remote sensing and lookup table
    split_df_dict = split_interim_dataframe(pd.read_parquet(input_path))
    
    click.echo('Compute summary statistics...')
    if compute_norm_stats:
        # compute norm stats
        norm_stats = _get_summary_statistics(
            density_col=split_df_dict['lookup_df']['Density'],
            tc_cols=split_df_dict['remote_sensing_df'][['TCW','TCG','TCB']]
        )
        # save to JSON file
        with norm_stats_path.open('w') as f:
            json.dump(norm_stats, f, indent=4)
        click.echo(f"Saved normalization statistics to {norm_stats_path}")
    else:
        # Load norm_stats from JSON
        with norm_stats_path.open('r') as f:
            norm_stats = json.load(f)
        click.echo(f"Loaded normalization statistics from {norm_stats_path}")
        
    click.echo('Processing sequence data...')
    # process and save time series data.
    process_and_save_sequences(
        **split_df_dict,
        seq_out_dir=output_seq_dir,
        lookup_out_path=output_lookup_path,
        norm_stats=norm_stats
    )
    click.echo(f'Saved lookup table to {output_lookup_path}')
    click.echo(f'Saved sequences to {output_seq_dir}')
    
if __name__ == '__main__':
    main()