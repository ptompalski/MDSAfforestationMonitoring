import pyreadr
import pandas as pd
import numpy as np
import click
import os

def load_data(file):
    """
    Loads an RDS file using pyreadr and returns the data as a pandas dataframe.
    
    Parameters
    ----------
    file : str
        Path to the .rds file.

    Returns
    -------
    pd.dataframe
        Loaded dataset.
    """
    result = pyreadr.read_r(file)
    
    return result[None]


@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help='Path to input RDS file.')
@click.option('--output_dir', type=click.Path(file_okay=False), required=True, help='Directory to save data as parquet file')
def main(input_path,output_dir):
    '''
    Command-line interface to load RDS data as a Parquet file for further processing.
    '''
    # get output path
    output_path = os.path.join(output_dir,'raw_data.parquet')
    
    print('Saving RDS file as Parquet file...')
    
    # load data and save to parquet
    load_data(input_path).to_parquet(output_path)

    print(f'File saved to {output_path}')
     
if __name__ == '__main__':
    main()
    