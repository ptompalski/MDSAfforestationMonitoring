import pandas as pd
import numpy as np
import click
import os

def data_split(df: pd.DataFrame, seed: int = 591, prop_train: float = 0.7):
    """
    Splits the data into training and testing sets by site.
    Each site is either in training set or in the test set. 

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    seed : int
        Random seed for reproducability.
    prop_train : float
        Proportion of sites to include in training sample (between 0 and 1):
        Default value: 0.7

    Returns
    -------
    pd.DataFrame
        Training data containing records from 70% of sites.
    pd.DataFrame
        Testing data containing records from the remaining 30% sites.
    """
    # check training proportion between 0 and 1
    if not (0 < prop_train < 1):
        raise ValueError("prop_train must be strictly between 0 and 1.")
    
    id_list = np.arange(df['ID'].min(), df['ID'].max()+1)
    # initialize split parameters
    n_training_sites = int(prop_train*len(id_list))
    np.random.seed(seed)
    # randomly select proportion of sites for training set
    training_ids = np.random.choice(id_list, size=n_training_sites, replace=False)

    # filter the data for site IDs selected in the training set
    df_train = df[df['ID'].isin(training_ids)]
    df_test = df[~df['ID'].isin(training_ids)]
    
    return df_train, df_test


@click.command()
@click.option('--input_path',type=click.Path(exists=True),required=True,help='Path to input data. Expecting parquet format.')
@click.option('--output_dir',type=click.Path(file_okay=False),required=True,help='Directory to save train/test data')
@click.option('--prop_train',
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=0.7,
    help="Proportion of data assigned to training sample, Default = 0.7."
)
@click.option('--seed', type=int, default=591, help="Random seed for reproducibility.")
def main(input_path,output_dir,prop_train,seed):
    '''
    Command-line interface for splitting data into train/test sets.
    '''
    
    print('\nSplitting data into train/test sets...')

    # Read in unsplit data
    # NOTE: I am assuming data is saved as parquet by this stage
    df_full = pd.read_parquet(input_path)
    
    # split into train and test sets
    df_train,df_test = data_split(df_full,seed,prop_train)
    
    print('Saving data...')
    
    # save training and testing data
    output_dir = os.path.join(output_dir, str(input_path.split('/')[-2]))
    os.makedirs(output_dir, exist_ok=True)
    output_path_train = os.path.join(output_dir, 'train_data.parquet')
    output_path_test = os.path.join(output_dir, 'test_data.parquet')
    df_train.to_parquet(output_path_train)
    print(f'Training data saved to {output_path_train}')
    
    df_test.to_parquet(output_path_test)
    print(f'Testing data saved to {output_path_test}')
    
if __name__ == '__main__':
    main()


