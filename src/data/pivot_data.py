import pandas as pd
import click
import gc
import os
from datetime import timedelta


def pivot_df(df):
    """
    Pivot the cleaned dataset to combine the survival rate columns into a single 'target' column 
    and match each survival rate records with its corresponding assessment date ('SrvvR_Date').  
    
    This function performs the following steps:
    1. Pivot survival rate columns ('SrvvR_1' to 'SrvvR_7') into the 'target' column, with a corresponding 'Age' column.
    2. Pivot the assessment date columns ('AsssD_1' to 'AsssD_7') into the 'SrvvR_Date' column, with a corresponding 'Age' column.
    3. Merge the two pivoted dataframes to match each record with its corresponding assessment date. 
    4. Filters out rows with out-of-range survival values (not between 0 to 100).
    
    Parameters
    ----------
        df : pd.DataFrame
            Cleaned DataFrame.

    Returns
    ----------
    pd.DataFrame
        Pivoted Data Frame with columns:
            - ID
            - PixelID
            - Density
            - Season
            - Type
            - Age
            - target: Survival Rate 
            - SrvvR_Date: Assessment date
    """
    # 1. Pivoting Survival Rate Columns
    df_sr = df[['ID', 'PixelID', 'Density', 'Season',  'Type', 
                'SrvvR_1', 'SrvvR_2', 'SrvvR_3', 'SrvvR_4', 
                'SrvvR_5', 'SrvvR_6', 'SrvvR_7']]

    df_sr.columns = ['ID', 'PixelID', 'Density','Season', 'Type', 
                     1, 2, 3, 4, 5, 6, 7]  # Rename survival rate columns for matching

    df_sr = df_sr.melt(
        id_vars=['ID', 'PixelID', 'Density', 'Type', 'Season'], 
        value_vars=[1, 2, 3, 4, 5, 6, 7],
        var_name='Age',
        value_name='target'
    ).dropna(axis=0, subset='target').drop_duplicates()

    # 2. Pivoting Assessment Date Columns
    df_ad = df[['ID', 'PixelID', 'AsssD_1', 'AsssD_2',
                'AsssD_3', 'AsssD_4', 'AsssD_5', 'AsssD_6', 'AsssD_7']]

    df_ad.columns = ['ID', 'PixelID', 1, 2, 3, 4, 5, 6, 7]  # Rename assessment date columns for matching

    df_ad = df_ad.melt(
        id_vars=['ID', 'PixelID'],
        value_vars=[1, 2, 3, 4, 5, 6, 7],
        var_name='Age', value_name='SrvvR_Date'
    ).dropna(axis=0, subset='SrvvR_Date').drop_duplicates()

    # 3. Matching Survival Rate with Assessment Date Column
    df = df_sr.merge(df_ad, on=['ID', 'PixelID', 'Age'])
    
    # 4. Remove out-of-range survival rate records.
    df = df[df['target'].between(0, 100, inclusive='both')]

    return df


# Matching Survey Records with VIs Signals
# Keep records with survival rate measurements but no satellite data.
def match_vi(df_pivot, df, day_range=16):
    """
    Calculate the average spectral signals within the specified time window
    around the assessment dates and match them to the corresponding survey record in the pivoted target dataframe. 

    For each row in the pivoted target dataset, this function:
    1. Defines a time window around the assessment date (+/- `day_range` days).
    2. Extract all satellite records from the cleaned dataframe that fall within the time window.
    3. Calculate the mean signal for each spectral index within the time window.
    4. Add the averages to the pivoted target dataframe.

    Parameters
    ----------
        df_pivot : pd.DataFrame
            The pivoted target dataframe.

        df : pd.DataFrame
            The cleaned dataframe.

        day_range : int, default=16
            Number of days before and after the assessment date to include when averaging.

    Returns
    ----------
    pd.DataFrame
        Pivoted target dataframe with the averaged spectral signals values around the assessment date.
    """

    cols = ['NDVI', 'SAVI', 'MSAVI', 'EVI',
            'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG', 'TCW']  # columns to average

    df = df[['ID', 'PixelID', 'ImgDate'] + cols].copy()
    df['ImgDate'] = df['ImgDate'].astype('datetime64[ns]').dt.date

    # 1. Compute time window for each survey record
    df_pivot = df_pivot.copy()
    df_pivot['SrvvR_Date'] = df_pivot['SrvvR_Date'].astype('datetime64[ns]').dt.date
    df_pivot['date_b'] = df_pivot['SrvvR_Date'] - timedelta(days=day_range)
    df_pivot['date_a'] = df_pivot['SrvvR_Date'] + timedelta(days=day_range)

    # 2. Extract all satellite records within the time window
    df = df.merge(df_pivot, on=['ID', 'PixelID'])
    df = df[df['ImgDate'].between(
        df['date_b'], df['date_a'], inclusive='both')]

    # 3. Calculate the average spectral signals.
    df_avg = df.groupby(['ID', 'PixelID', 'Age'])[cols].mean().reset_index()

    # 4. Add the average signals to its corresponding records in the pivoted dataframe.
    df_match = df_pivot.merge(df_avg, on=['ID', 'PixelID', 'Age'], how='left')

    return df_match.drop(columns=['date_b', 'date_a'])


def target_to_bin(df, threshold=None):
    """
    Map target to binary class "High"/"Low" survival rate based on the given threshold.

    Parameters
    ----------
        df : pd.DataFrame
            Pivoted DataFrame.

        threshold : float, default=None 
            Survival rate classification threshold. Must be a value between 0 to 1.

    Returns
    ----------
    pd.DataFrame
        Pivoted dataframe with the target column mapped to binary class 'High'/'Low'.
    
    Raises
    ------
    ValueError
        If threshold is None or not in [0, 1].
    """
    if threshold is None:
        raise ValueError("Threshold for classifying survival rate required. Please enter a value between 0 and 1.")
    if not (0 <= threshold <=1) :
        raise ValueError(f"The threshold must be between 0 and 1, got {threshold}.")
    df = df.copy()
    df['target'] = (df['target'] < threshold*100).map(
        {True: 'Low', False: 'High'})
    
    return df


@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help='Path to cleaned data. Expecting parquet format.')
@click.option('--output_dir', type=click.Path(file_okay=False), required=True, help='Directory to save pivoted data')
@click.option('--day_range', required=True, type=click.IntRange(0, 182), help='Number of days before and after the assessment date to include when averaging.')
@click.option('--threshold', required=True, type=click.FloatRange(0, 1), help='Survival Rate Threshold, between 0 to 1.')
def main(input_path, output_dir, day_range, threshold):
    '''
    Command-line interface for pivoting and mapping target columns.
    Preprocessing target will be performed in following steps. 
    '''
    # Load Data
    print('Loading cleaned data...')
    df = pd.read_parquet(input_path)
    
    # Pivoting Data
    print('Pivoting target features ...')
    df_target = pivot_df(df)
    
    # Target Feature Matching
    print('Matching image date with assessment date ...')
    df_matched = match_vi(df_target, df, day_range)

    # Target to Binary
    print('Matching target to binary classes...')
    df_matched_bin = target_to_bin(df_matched, threshold)
    
    del df, df_target, df_matched       
    gc.collect()         
    
    # Saving Preprocessed DataFrame
    print('Saving preprocessed dataset...')
    output_path_processed = os.path.join(
        output_dir, f'processed_data{threshold*100}.parquet')
    df_matched_bin.to_parquet(output_path_processed)

    print(f'target data saved to {output_path_processed}')


if __name__ == '__main__':
    main()
