import pytest
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.get_time_series import (
    _get_summary_statistics, process_single_site,
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
    
@pytest.fixture
def sample_lookup_data():
    '''
    Create sample lookup table for testing
    '''
    lookup_df = pd.DataFrame({
        'ID': [1,2,3],
        'PixelID': ['1_101','2_102','3_103'],
        'SrvvR_Date': pd.to_datetime(['2020-06-01']*3),
        'Age': [5,6,7],
        'Density': [1200,1500,1800],
        'Type': ['Mixed','Conifer','Decidous'],
        'target': [70.0,80.0,90.0]
    })
    return lookup_df
    
@pytest.fixture
def sample_remote_sensing_data():
    '''
    Create sample remote sensing table for testing.
    '''
    remote_df = pd.DataFrame({
        'ID': [1,1,1,1,2,3],
        'PixelID': ['1_102','1_101','1_101','1_101','2_102','3_103'],
        'ImgDate': pd.to_datetime(['2019-01-01','2020-05-01','2020-06-01','2020-06-02','2024-01-01','2019-01-01']),
        'NDVI':  [0.3,0.3,0.3,0.3,0.3,0.3],
        'SAVI':  [0.3,0.3,0.3,0.3,0.3,0.3],
        'MSAVI': [0.3,0.3,0.3,0.3,0.3,0.3],
        'EVI':   [0.3,0.3,0.3,0.3,0.3,0.3],
        'EVI2':  [0.3,0.3,0.3,0.3,0.3,0.3],
        'NDWI':  [0.3,0.3,0.3,0.3,0.3,0.3],
        'NBR':   [0.3,0.3,0.3,0.3,0.3,0.3],
        'DOY': [1, 122, 153, 154,1,1],
        'TCW': [14,16,18,20,22,24],  # (14,16,18,20,22,24] - 10) / 2 = [2.0,3.0,4.0,5.0,6.0,7.0]
        'TCG': [16,18,20,22,24,26],  # ([[16,18,20,22,24,26] - 20) / 2 = [-2.0,-1.0,0,1.0,2.0,3.0]
        'TCB': [33,39,45,51,57,63],   # ([33,39,45,51,57,63] - 30) / 3 = [1.0,3.0,5.0,7.0,9.0,11.0]
        'NDVI': [0.3,0.3,0.3,0.3,0.3,0.3]
    })
    return remote_df
    
@pytest.fixture()
def sample_norm_stats():
    '''
    Create sample normalization statistics dictionary for testing
    '''
    norm_stats = {
        'mean': {'TCW': 10, 'TCG': 20, 'TCB': 30, 'Density': 1000},
        'std': {'TCW': 2, 'TCG': 2, 'TCB': 3, 'Density': 100}
    }
    return norm_stats

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

def test_process_single_site_with_records(sample_lookup_data,sample_remote_sensing_data):
    '''
    Test to ensure records are filtered and normalized correctly if sequences exist for a site
    '''
    #Setup temporary output paths
    tmp_output_dir = Path('tmp')
    tmp_output_dir.mkdir(exist_ok=True)
    
    # prepare input for process_single_site
    row_with_sequences = sample_lookup_data.iloc[0]
    test_group_df = sample_remote_sensing_data.query("ID == 1 and PixelID == '1_101'")
    
    # get output
    output_row_idx, output_filename = process_single_site(
        row=(0,row_with_sequences),
        group_df=test_group_df,
        seq_out_dir=tmp_output_dir
    )
    
    # check correctness of output
    assert output_row_idx == 0
    assert output_filename == '1_1_101_2020-06-01.parquet'
    
    # check that file was saved
    assert os.path.exists(tmp_output_dir/'1_1_101_2020-06-01.parquet')
    
    # check filtering and time feature engineering
    test_seq_data = pd.read_parquet(tmp_output_dir/'1_1_101_2020-06-01.parquet')
    assert len(test_seq_data) == 2
    assert np.allclose(
        test_seq_data['neg_cos_DOY'].to_numpy(),
        -np.cos(2 * np.pi * np.array([122, 153]) / 365)
    )
    assert np.allclose(
        test_seq_data['log_dt'].to_numpy(),
        np.log1p(np.array([31, 0]))
    )
    
    # check columns are correctly dropped
    assert set(test_seq_data.columns) == {
        'NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR', 
        'TCB', 'TCG', 'TCW','neg_cos_DOY','log_dt'
    }
    
    # get rid of tmp directory once finished 
    shutil.rmtree(tmp_output_dir)
    
def test_process_single_site_no_records(sample_lookup_data,sample_remote_sensing_data):
    '''
    Test to ensure process_single_site correctly handles lookup rows with no valid sequence data.
    '''
     #Setup temporary output paths
    tmp_output_dir = Path('tmp')
    tmp_output_dir.mkdir(exist_ok=True)
    
    # obtain a row and data with remote sensing data outside of time range
    row_no_sequences = sample_lookup_data.iloc[1]
    test_group_df = sample_remote_sensing_data.query("ID == 2 and PixelID == '2_102'")
    
    # get output, should be None as the matching record is out of time range
    output = process_single_site(
        row=(1,row_no_sequences),
        group_df=test_group_df,
        seq_out_dir=tmp_output_dir
    )
    assert output is None
    
    # get rid of tmp directory once finished 
    shutil.rmtree(tmp_output_dir)

def test_process_and_save_sequences():
    '''
    Test to ensure sequence files are saved properly and engineering is correct
    '''
    pass

    # # Setup temporary output paths
    # tmp_output_dir = Path('tmp')
    # tmp_output_dir.mkdir(exist_ok=True)
    
    # tmp_output_lookup_path = tmp_output_dir/ 'tmp_lookup.parquet'
    
    # # Call function
    # process_and_save_sequences(
    #     lookup_df=lookup_df,
    #     remote_sensing_df=remote_df,
    #     seq_out_dir=tmp_output_dir,
    #     lookup_out_path=tmp_output_lookup_path,
    #     norm_stats=norm_stats
    # )

    # # Verify files exist, lookup and sequence.
    # saved_files = list(tmp_output_dir.glob("*.parquet"))
    # assert len(saved_files) == 2

    # # check columns and feature engineering
    # lookup_df_out = pd.read_parquet(tmp_output_lookup_path)
    # assert 'filename' in lookup_df_out.columns
    # assert np.isclose(lookup_df_out['Density'].iloc[0], 2.0)  # (1200 - 1000)/100

    # # Read the sequence file and test engineering
    # sequence_df_out = pd.read_parquet(saved_files[0])
    # assert np.isclose(sequence_df_out['log_dt'].iloc[0], np.log1p(152))
    # assert np.isclose(sequence_df_out['neg_cos_DOY'].iloc[0], -np.cos(2 * np.pi * 1 / 365), atol=1e-6)
    # assert np.isclose(sequence_df_out['TCW'].iloc[0], 2.0)
    # assert np.isclose(sequence_df_out['TCG'].iloc[0], -2.0)
    # assert np.isclose(sequence_df_out['TCB'].iloc[0], 1.5)
    
    
    # # get rid of tmp directory once finished 
    # shutil.rmtree(tmp_output_dir)