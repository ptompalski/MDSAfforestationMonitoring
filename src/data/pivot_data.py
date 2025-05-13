import pandas as pd
import click
import gc
import os
from datetime import timedelta

def pivot_df(df):
    """
    Pivot data to combine Survival Rates and Assessment Dates into two separate columns. 
    
    Parameters
    ----------
        df : pd.DataFrame
            Original DataFrame.
    
    Returns
    ----------
    pd.DataFrame
        Pivoted Data Frame
    """
    # Pivoting Survival Rate Columns
    df_sr = df[['ID', 'PixelID', 'Density', 'Season',  'Type', 'SrvvR_1',
                'SrvvR_2', 'SrvvR_3', 'SrvvR_4', 'SrvvR_5', 'SrvvR_6', 'SrvvR_7']]
    df_sr.columns = ['ID', 'PixelID', 'Density',
                  'Season', 'Type', 1, 2, 3, 4, 5, 6, 7]
    
    df_sr = df_sr.melt(
        id_vars=['ID', 'PixelID', 'Density', 'Type', 'Season'], value_vars=[1, 2, 3, 4, 5, 6, 7],
        var_name='Age',
        value_name='target'
    ).dropna(axis=0, subset='target').drop_duplicates()

    # Pivoting Assessment Date Columns
    df_ad = df[['ID', 'PixelID', 'AsssD_1', 'AsssD_2',
                'AsssD_3', 'AsssD_4', 'AsssD_5', 'AsssD_6', 'AsssD_7']]

    df_ad.columns = ['ID', 'PixelID', 1, 2, 3, 4, 5, 6, 7]
    
    df_ad = df_ad.melt(
        id_vars=['ID', 'PixelID'],
        value_vars=[1, 2, 3, 4, 5, 6, 7],
        var_name='Age', value_name='SrvvR_Date'
    ).dropna(axis=0, subset='SrvvR_Date').drop_duplicates()
    
    # Matching Survival Rate with Assessment Date Column
    df = df_sr.merge(df_ad, on=['ID', 'PixelID', 'Age'])
    df = df[df['target'].between(0, 100, inclusive='both')]
    
    float_cols = ['target', 'Density']
    int_cols = ['Season', 'Age', 'ID'] 
    df[int_cols] = df[int_cols].astype('int32')
    df[float_cols] = df[float_cols].astype('float32')
    df['SrvvR_Date'] = df['SrvvR_Date'].astype('datetime64[ns]')
    
    return df


# Matching Survey Records with VIs Signals
# Keep records with survival rate measurements but no satellite data.
def mean_vi(df_pivot, df, cols):
    """
    Calculate mean VI signal: average of ± specified window of Assessment Date.

    Parameters
    ----------
        df_melt : pd.DataFrame
            Single row from the pivoted DataFrame.

        df : pd.DataFrame
            Original DataFrame

        day_range : int
            Averaging window / 2, range = 1 to 182

        cols : list
            List of spectral indices to average.

    Returns
    ----------
    pd.Series
        Averaged spectral indices as a series.
    """
    try:
        df = df.loc[[(df_pivot['ID'], df_pivot['PixelID'])]]
        df = df[df['ImgDate'].between(
            df_pivot['date_b'], df_pivot['date_a'], inclusive='both')]
        return df[cols].mean().astype('float32')
    except KeyError:
        return pd.Series([None] * len(cols), index=cols)


def match_vi(df_pivot, df, day_range):
    """
    Match mean vi signal to pixel by Assessment Date.

    Parameters
    ----------
        df_melt : pd.DataFrame
            Single row from the pivoted DataFrame.

        df : pd.DataFrame
            Original DataFrame

        day_range : int
            Averaging window / 2, range = 0 to 182

    Returns
    ----------
    pd.DataFrame
        Pivoted Target dataframe with averaged signal value.
    """
    cols = ['NDVI', 'SAVI', 'MSAVI', 'EVI',
            'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG', 'TCW']

    df = df.set_index(['ID', 'PixelID'])
    df = df[cols + ['ImgDate']]
    df.loc[:, 'ImgDate'] = df['ImgDate'].astype('datetime64[ns]').dt.date
    df.loc[:, cols] = df[cols].astype('float32')
    td = timedelta(days=day_range)
    df_pivot.loc[:, 'date_b'] = df_pivot['SrvvR_Date'] - td
    df_pivot.loc[:, 'date_a'] = df_pivot['SrvvR_Date'] + td
    df_pivot.loc[:, cols] = df_pivot.apply(
        lambda x: mean_vi(x, df, cols), axis=1)
    return df_pivot.drop(columns=['date_b', 'date_a'])


def target_to_bin(df, threshold=None):
    """
    Map target to binary class "High"/"Low" survival rate base on given threshold.
    
    Parameters
    ----------
        df : pd.DataFrame
            Pivoted DataFrame.
        
        threshold : float
            Survival Rate Threshold. 0 to 100
    
    Returns
    ----------
    pd.DataFrame
        Original dataframe with the target column mapped to binary class 'High'/'Low'.
    """
    if threshold is None:
        raise ValueError(
            "Threshold for classifying survival rate required. Please enter a value between 0 and 100.")

    df['target'] = (df['target'] < threshold).map(
        {True: 'Low', False: 'High'})
    return df


@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help='Path to raw input data. Expecting parquet format.')
@click.option('--output_dir', type=click.Path(file_okay=False), required=True, help='Directory to save pivoted data')
@click.option('--day_range', required=True, type=click.IntRange(0, 182), help='Averaging Window')
@click.option('--threshold', required=True, type=click.FloatRange(0, 1), help='Survival Rate Threshold')
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

    # Saving Preprocessed DataFrame
    print('Saving preprocessed dataset...')
    output_path_processed = os.path.join(
        output_dir, f'processed_data{threshold*100}.parquet')
    df_matched_bin.to_parquet(output_path_processed)

    print(f'target data saved to {output_path_processed}')


if __name__ == '__main__':
    main()
