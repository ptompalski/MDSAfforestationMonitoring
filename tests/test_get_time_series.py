import pytest
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.get_time_series import (
    _get_summary_statistics, _get_raw_sequence,
    split_interim_dataframe, process_and_save_sequences
)

@pytest.fixture()
def sample_interim_data():
    '''
    Sample interim dataframe for testing split function.
    '''
    interim_df = pd.DataFrame({
    'ID': [1, 2, 2],
    "PixelID": ['1_1', '2_2', '2_2'],
    "Season": [2001, 2002, 2002],
    'Type': ['Conifer', 'Mixed', 'Mixed'],
    "Density": [1434.54, 1235, 1235],
    'SrvvR_1': [None, 90, 90],
    'SrvvR_2': [80, 90, 90],
    'SrvvR_3': [None, None, None],
    'SrvvR_4': [70, 90, 90],
    'SrvvR_5': [None, 90, 90],
    'SrvvR_6': [60, None, None],
    'SrvvR_7': [101, None, None],
    'AsssD_1': [None, '2009-07-01', '2009-07-01'],
    'AsssD_2': ['2010-07-01']*3,
    'AsssD_3': [None, None, None],
    'AsssD_4': ['2012-07-01']*3,
    'AsssD_5': [None, '2013-07-01', '2013-07-01'],
    'AsssD_6': ['2014-07-01', None, None],
    'AsssD_7': ['2015-07-01', None, None],
    'ImgDate': ['2001-01-01', '2002-01-01', '2003-01-01'],
    'DOY':[100,200,300],
    'NDVI':  [0.3]*3,
    'SAVI':  [0.3]*3,
    'MSAVI': [0.3]*3,
    'EVI':   [0.3]*3,
    'EVI2':  [0.3]*3,
    'NDWI':  [0.3]*3,
    'NBR':   [0.3]*3,
    'TCB':   [0.3]*3,
    'TCG':   [0.3]*3,
    'TCW':   [0.3]*3
    })
    
    return interim_df
    

def test_get_summary_statistics():
    '''
    Test to ensure summary statistics for Density, TCB, TCG, TCW are correctly computed.
    '''
    # sample input
    sample_density_col = pd.Series([1000,2000,6000,600,400])
    sample_tc_cols = pd.DataFrame({
        'TCB': [10,20,30,-10,-20,-30],
        'TCG': [10,30,-5,25,0,0],
        'TCW': [5,5,5,5,5,5]
    })
    
    # expected output
    expected_output = {
        'mean': {'TCB': 0, 'TCG': 10.0, 'TCW': 5, 'Density': 2000.0},
        'std':  {'TCB': 23.664, 'TCG': 14.491, 'TCW': 0, 'Density': 2319.483}
    }
    
    # actual output
    output = _get_summary_statistics(sample_density_col,sample_tc_cols)
    
    # test dict properties
    assert output.keys() == expected_output.keys()
    assert output['mean'].keys() == expected_output['mean'].keys()
    assert output['std'].keys() == expected_output['std'].keys()
    
    # test dict mean values
    assert np.isclose(
        list(output['mean'].values()), 
        list(expected_output['mean'].values()), 
        atol=1e-03
        ).all()
    
    # test dict std values
    assert np.isclose(
        list(output['std'].values()), 
        list(expected_output['std'].values()), 
        atol=1e-03
        ).all()
    
def test_split_interim_datafram(sample_interim_data):
    '''
    Test to ensure correct splitting of interim dataframe into lookup table and remote sensing table
    '''
    expected_lookup_df = pd.DataFrame({
        'ID':[1]*3 + [2]*4,
        'PixelID':['1_1']*3 + ['2_2']*4,
        'SrvvR_Date':['2010-07-01','2012-07-01','2014-07-01','2009-07-01','2010-07-01','2012-07-01','2013-07-01'],
        'Age':[2,4,6,1,2,4,5],
        'Density':[1434.54]*3 + [1235.00]*4,
        'Type':['Conifer']*3 + ['Mixed']*4,
        'target':[80.0,70.0,60.0] + [90.0]*4
    })
    expected_lookup_df['SrvvR_Date'] = pd.to_datetime(expected_lookup_df['SrvvR_Date'])
    
    expected_remote_sensing_df = pd.DataFrame({
        'ID':[1,2,2],
        'PixelID':['1_1','2_2','2_2'],
        'ImgDate':['2001-01-01','2002-01-01','2003-01-01'],
        'DOY':[100,200,300],
        'NDVI':  [0.3]*3,
        'SAVI':  [0.3]*3,
        'MSAVI': [0.3]*3,
        'EVI':   [0.3]*3,
        'EVI2':  [0.3]*3,
        'NDWI':  [0.3]*3,
        'NBR':   [0.3]*3,
        'TCB':   [0.3]*3,
        'TCG':   [0.3]*3,
        'TCW':   [0.3]*3
    })
    expected_remote_sensing_df['ImgDate'] = pd.to_datetime(expected_remote_sensing_df['ImgDate'])
    
    results = split_interim_dataframe(sample_interim_data)
    
    # test lookup df
    assert (
        results['lookup_df']
        .sort_values(by='SrvvR_Date')
        .reset_index(drop=True)
        .equals(
            expected_lookup_df
            .sort_values(by='SrvvR_Date')
            .reset_index(drop=True)
        )
    )
    
    # test remote sensing df
    assert (
        results['remote_sensing_df']
        .sort_values(by='ImgDate')
        .reset_index(drop=True)
        .equals(
            expected_remote_sensing_df
            .sort_values(by='ImgDate')
            .reset_index(drop=True)
        )
    )
    
def test_get_raw_sequence_filters_and_sorts():
    '''
    Test to ensure get_raw_sequence filters and sorts correctly
    '''
    df = pd.DataFrame({
        'ID': [1, 1, 1, 2],
        'PixelID': [101, 101, 101, 102],
        'ImgDate': pd.to_datetime(['2020-01-01', '2020-06-01', '2021-01-01', '2020-01-01']),
        'DOY': [1, 153, 1, 1],
        'NDVI': [0.5, 0.6, 0.7, 0.8]
    })

    site_key = {
        'ID': 1,
        'PixelID': 101,
        'SrvvR_Date': pd.Timestamp('2020-12-31')
    }

    result = _get_raw_sequence(site_key, df)

    expected = df.iloc[[0, 1]].sort_values(by="ImgDate").reset_index(drop=True)
    assert result.reset_index(drop=True).equals(expected)

def test_get_raw_sequence_returns_none_for_no_match():
    '''
    Test to ensure _get_raw_sequence can handle lookups with no matching sequences.
    '''
    df = pd.DataFrame({
        'ID': [2],
        'PixelID': [102],
        'ImgDate': pd.to_datetime(['2020-01-01']),
        'DOY': [1],
        'NDVI': [0.8]
    })

    site_key = {
        'ID': 1,
        'PixelID': 101,
        'SrvvR_Date': pd.Timestamp('2021-01-01')
    }

    assert _get_raw_sequence(site_key, df) is None

def test_process_and_save_sequences():
    '''
    Test to ensure sequence files are saved properly and engineering is correct
    '''
    
    # create mock data
    lookup_df = pd.DataFrame({
        'ID': [1],
        'PixelID': [101],
        'SrvvR_Date': pd.to_datetime(['2020-06-01']),
        'Age': [5],
        'Density': [1200],
        'Type': ['Mixed'],
        'target': [1]
    })

    remote_df = pd.DataFrame({
        'ID': [1],
        'PixelID': [101],
        'ImgDate': pd.to_datetime(['2020-01-01']),
        'NDVI':  [0.3],
        'SAVI':  [0.3],
        'MSAVI': [0.3],
        'EVI':   [0.3],
        'EVI2':  [0.3],
        'NDWI':  [0.3],
        'NBR':   [0.3],
        'DOY': [1],
        'TCW': [14],  # (14 - 10) / 2 = 2.0
        'TCG': [16],  # (16 - 20) / 2 = -2.0
        'TCB': [33],   # (33 - 30) / 2 = 1.5
        'NDVI': [0.6]
    })

    # mock norm stats
    norm_stats = {
        'mean': {'TCW': 10, 'TCG': 20, 'TCB': 30, 'Density': 1000},
        'std': {'TCW': 2, 'TCG': 2, 'TCB': 2, 'Density': 100}
    }

    # Setup temporary output paths
    tmp_output_dir = Path('tmp')
    tmp_output_dir.mkdir(exist_ok=True)
    
    tmp_output_lookup_path = tmp_output_dir/ 'tmp_lookup.parquet'
    
    # Call function
    process_and_save_sequences(
        lookup_df=lookup_df,
        remote_sensing_df=remote_df,
        seq_out_dir=tmp_output_dir,
        lookup_out_path=tmp_output_lookup_path,
        norm_stats=norm_stats
    )

    # Verify files exist, lookup and sequence.
    saved_files = list(tmp_output_dir.glob("*.parquet"))
    assert len(saved_files) == 2

    # check columns and feature engineering
    lookup_df_out = pd.read_parquet(tmp_output_lookup_path)
    assert 'filename' in lookup_df_out.columns
    assert np.isclose(lookup_df_out['Density'].iloc[0], 2.0)  # (1200 - 1000)/100

    # Read the sequence file and test engineering
    sequence_df_out = pd.read_parquet(saved_files[0])
    assert np.isclose(sequence_df_out['log_dt'].iloc[0], np.log1p(152))
    assert np.isclose(sequence_df_out['neg_cos_DOY'].iloc[0], -np.cos(2 * np.pi * 1 / 365), atol=1e-6)
    assert np.isclose(sequence_df_out['TCW'].iloc[0], 2.0)
    assert np.isclose(sequence_df_out['TCG'].iloc[0], -2.0)
    assert np.isclose(sequence_df_out['TCB'].iloc[0], 1.5)
    
    
    # get rid of tmp directory once finished 
    shutil.rmtree(tmp_output_dir)