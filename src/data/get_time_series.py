import numpy as np
import pandas as pd
import os
import json
from src.data.pivot_data import pivot_df
import click
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Tuple, Union
import sys

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


def process_single_site(
    row: Tuple[int, pd.Series],
    group_df: pd.DataFrame,
    seq_out_dir: Path
) -> Union[Tuple[int, str], None]:
    """
    Process a single site record and generate a remote sensing sequence file.

    For the given survival record (by ID, PixelID, SrvvR_Date), this function:
    - Filters the remote sensing dataset up to the survey date
    - Computes time-based features (log time delta, cosine-transformed DOY)
    - Saves the sequential data as a Parquet file to the specified output directory

    Parameters
    ----------
    row : Tuple[int, pd.Series]
        A tuple where:
        - The first element is the index of the row in the original lookup DataFrame
        - The second element is the row itself as a pandas Series.

    group_df : pd.DataFrame
        partition of remote sensing dataset containing at least the following columns:
        'ID', 'PixelID', 'ImgDate', 'DOY', and vegetation indices including 'TCW', 'TCG', 'TCB', etc.
        The dataframe must only contain remote sensing records with ID and PixelID matching that of
        the given lookup row.

    seq_out_dir : Path
        Directory where the generated sequence Parquet file should be saved.
        Files are stored with the naming convention <ID><PixelID><SrvvR_Date>.parquet,
        where SrvvR_Date is the survey date of the target survival rate.

    Returns
    -------
    Union[Tuple[int, str], None]
        Returns (row index, filename) if a sequence was successfully created,
        otherwise returns None if no matching remote sensing records were found.
    """
    idx, row = row
    record_date = row['SrvvR_Date']    
              
    # Filter by time      
    time_filter = group_df["ImgDate"] <= record_date
    sequence_df = group_df[time_filter].copy()
    
    # if no matching rows, return None 
    if sequence_df.empty:
        return None
    
    # Process time-based features
    time_delta = (record_date - sequence_df["ImgDate"]).dt.days
    sequence_df["log_dt"] = np.log1p(time_delta)
    sequence_df["neg_cos_DOY"] = -np.cos(2 * np.pi * sequence_df["DOY"] / 365)
    sequence_df = sequence_df.drop(columns=["DOY"])

    # get filename and path to store sequence data
    fname = f"{row["ID"]}_{row["PixelID"]}_{record_date.strftime('%Y-%m-%d')}.parquet"
    out_path = seq_out_dir/fname

    # drop uneeded columns: ID, PixelID, ImgDate before saving and arrange by log dt in descending order (earliest to latest records)
    sequence_df = sequence_df.drop(columns=['ID', 'PixelID', 'ImgDate']).sort_values(by='log_dt',ascending=False)
    
    # save file
    sequence_df.to_parquet(out_path, index=False)

    return idx, fname
    
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
    'ID','PixelID','ImgDate','DOY',
    'NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG', 'TCW' 
    ]
    
    lookup_cols = [
        'ID','PixelID','SrvvR_Date','Age','Density','Type','target'
    ]
    
    # get lookup table and remove duplicate rows if they exist (but they shouldn't), and convert to datetime
    lookup_df = pivot_df(interim_df).drop_duplicates()[lookup_cols].reset_index(drop=True)
    lookup_df['SrvvR_Date'] = pd.to_datetime(lookup_df['SrvvR_Date'])
    
    # type conversion for consistency
    lookup_df['target'] = lookup_df['target'].astype(float)
    lookup_df['Age'] = lookup_df['Age'].astype(int)
    
    # some of the remote sensing cols are duplicated, should be dropped. Also convert ImgDate to datetime
    remote_sensing_df = interim_df[remote_sensing_cols].drop_duplicates().reset_index(drop=True)
    remote_sensing_df['ImgDate'] = pd.to_datetime(remote_sensing_df['ImgDate'])
    
    return {
        'lookup_df':lookup_df,
        'remote_sensing_df': remote_sensing_df
    }
    

def process_and_save_sequences(
    lookup_df: pd.DataFrame,
    remote_sensing_df: pd.DataFrame,
    seq_out_dir: str,
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
        Remote sensing DataFrame with columns:
        'ID', 'PixelID', 'ImgDate', 'DOY', and vegetation features such as 
        'NDVI', 'NDWI', 'EVI', 'SAVI', 'MSAVI', 'TCW', 'TCG', 'TCB', 'NBR', etc.

    seq_out_dir : str
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
        - Parquet file per record: '<seq_out_dir>/<ID>_<PixelID>_<SrvvR_Date>.parquet'
        - Lookup table: e.g., 'lookup_train.parquet' with columns:
            * ID, PixelID, SrvvR_Date, Age, Density, target
            * One-hot columns: Type_Conifer, Type_Decidous ('Mixed type treated as reference category')

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
    
    # group by site for faster access
    rs_grouped = remote_sensing_df.groupby(["ID", "PixelID"])
    
    # set up multiprocessing: pass idx,row, and matching group
    args = []
    for idx, row in lookup_df.iterrows():
        site_key = (row["ID"], row["PixelID"])
        if site_key in rs_grouped.indices:
            group_df = rs_grouped.get_group(site_key)
            args.append(((idx, row), group_df))

    # wrapper to only expose row and group as input
    func = partial(process_single_site, seq_out_dir=seq_out_dir)
    
    # use multiprocessing to get sequences for each row in lookup table
    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.starmap(func, args), 
                total=len(args), 
                desc="Processing sequences",
                file=sys.stdout
                )
            )

    # filter lookup table rows with no rows
    valid_results = [res for res in results if res is not None]
    valid_indices, fnames = zip(*valid_results) if valid_results else ([], [])
    filtered_lookup = lookup_df.loc[list(valid_indices)].copy().reset_index(drop=True)
    
    # normalize density column
    filtered_lookup["Density"] = (
        filtered_lookup["Density"] - norm_stats["mean"]["Density"]
    ) / norm_stats["std"]["Density"]

    # One-hot encoding of Type
    ohe_type = pd.get_dummies(filtered_lookup["Type"], prefix="Type", dtype=int).drop(columns=["Type_Mixed"], errors="ignore")
    filtered_lookup = pd.concat([filtered_lookup.drop(columns="Type"), ohe_type], axis=1)

    # add filename column to lookup table
    filtered_lookup["filename"] = pd.Series(fnames).reset_index(drop=True)
    filtered_lookup.to_parquet(lookup_out_path, index=False)
      
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