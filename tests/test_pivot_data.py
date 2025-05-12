from ..src.data.target_to_binary import target_to_bin
import pytest
import pandas as pd
import numpy as np
from ..src.data.pivot_data import pivot_df, mean_vi, match_vi, target_to_bin

# Test data for pivot_df
test_df = pd.DataFrame({
    'ID': [1, 2, 2],
    "PixelID": ['1_1', '2_2', '2_2'],
    "Season": [2001, 2002, 2002],
    'Type': ['Conifer', 'Mixed', 'Mixed'],
    "Density": [1434.54, 1235, 1235],
    'SrvvR_1': [None, 90, 90],
    'SrvvR_2': [90, 90, 90],
    'SrvvR_3': [None, None, None],
    'SrvvR_4': [90, 90, 90],
    'SrvvR_5': [None, 90, 90],
    'SrvvR_6': [90, None, None],
    'SrvvR_7': [101, None, None],
    'AssD_1': [None, '2009-07-01', '2009-07-01'],
    'AssD_2': ['2010-07-01']*3,
    'AssD_3': [None, None, None],
    'AssD_4': ['2012-07-01']*3,
    'AssD_5': [None, '2013-07-01', '2013-07-01'],
    'AssD_6': ['2014-07-01', None, None],
    'AssD_7': ['2015-07-01', None, None],
    'Year': [2001, 2002, 2003],
    'DOY': [1]*3,
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

exp_pivot_cols = ['ID', 'PixelID', 'Density', 'Type',
           'Season', 'Age', 'target', 'SrvvR_Date']

exp_pivot_df = pd.DataFrame(
    {
        'ID': [1, 1, 1, 2, 2, 2, 2],
        'PixelID': ['1_1', '1_1', '1_1', '2_2', '2_2', '2_2', '2_2'],
        'Density': [1434.54, 1434.54,  1434.54, 1235.0, 1235.0, 1235.0, 1235.0],
        'Type':  ['Conifer', 'Conifer',  'Conifer', 'Mixed', 'Mixed', 'Mixed', 'Mixed'],
        'Season': [2001, 2001, 2001, 2002, 2002, 2002, 2002],
        'Age': [2, 4, 6, 1, 2, 4, 5],
        'target': [90.0]*7,
        'SrvvR_Date': ['2010-07-01', '2012-07-01', '2014-07-01',  '2009-07-01', '2010-07-01', '2012-07-01', '2013-07-01']
    }
)
exp_pivot_df['Age'] = exp_pivot_df['Age'].astype('int32')
exp_pivot_df['target'] = exp_pivot_df['target'].astype('float32')

# Expected outlier to exclude
df_outlier = pd.Series({
    'ID': [1],
    'PixelID': ['1_1'],
    'Density': [1434.54],
    'Type':  ['Conifer'],
    'Season': [2001],
    'Age': [7],
    'target': [101],
    'SrvvR_Date': ['2015-07-01']
})

# Test Data for target_to_bin()
df_bin = pd.DataFrame(
    {
        'ID': [1, 1, 2, 3, 4, 5],
        'PixelID': ['123_321', '123_321', '123_321', '123_321', '123_321', '123_321'],
        'Density': [1434.54, 1235, 3415.43, 3413.15, 2458.43, 13463],
        'Type':  ['Conifer', 'Conifer', 'Mixed', 'Mixed', 'Decidous', 'Decidous'],
        'Season': ['2025', '2025', '2025', '2025', '2024', '2015'],
        'Age': ['1', '2', '3', '4', '5', '6'],
        'target': ['100', '95.6', '80.0', '74.6', '40.3', '0.0'],
        'SrvvR_Date': ['2009-07-01', '2009-07-01', '2009-07-01', '2009-07-01', '2009-07-01', '2009-07-01']
    }
)


def test_pivot_df():
    '''
    Test for if pivot_df() function correctly converts the target column into binary classes according to provided threshold.
    '''
    pivoted_df = pivot_df(test_df)
    
    assert pivoted_df.columns.to_list() == exp_pivot_cols
    assert len(pivoted_df) == len(exp_pivot_df)
    assert not pivoted_df.duplicated().all()
    assert pivoted_df.sort_values(
        by=['ID', 'Age']).reset_index(drop=True).equals(exp_pivot_df)
    assert not (pivoted_df == df_outlier).all(axis=1).any()
    assert not pivoted_df.isna().all()
    assert pivoted_df['Age'].dtype == 'int64'
    assert pivoted_df['target'].dtype == 'float32'
    assert pivoted_df['SrvvR_Date'].dtype == 'datetime64[ns]'


exp_match_df = pd.DataFrame(
    {
        'ID': [1]*4,
        'PixelID': ['1_1']*4,
        'Density': [1434.54]*4,
        'Type':  ['Conifer']*4,
        'Season': [2008]*4,
        'Age': [2, 4, 6, 7],
        'target': [90.0]*4,
        'SrvvR_Date': ['2010-07-01', '2012-07-01', '2014-07-01', '2015-07-01'],
        'NDVI':  [0.25, 0.75, 0.79, None],
        'SAVI':  [0.25, 0.75, 0.79, None],
        'MSAVI': [0.25, 0.75, 0.79, None],
        'EVI':   [0.25, 0.75, 0.79, None],
        'EVI2':  [0.25, 0.75, 0.79, None],
        'NDWI':  [0.25, 0.75, 0.79, None],
        'NBR':   [0.25, 0.75, 0.79, None],
        'TCB':   [0.25, 0.75, 0.79, None],
        'TCG':   [0.25, 0.75, 0.79, None],
        'TCW':   [0.25, 0.75, 0.79, None]
    }
)

pivoted_df = pd.DataFrame(
    {
        'ID': [1]*4,
        'PixelID': ['1_1']*4,
        'Density': [1434.54]*4,
        'Type':  ['Conifer']*4,
        'Season': [2008]*4,
        'Age': [2, 4, 6, 7],
        'target': [90.0]*4,
        'SrvvR_Date': ['2010-07-01', '2012-07-01', '2014-07-01', '2015-07-01'],
        'NDVI':  [0.25, 0.75, 0.79, None],
        'SAVI':  [0.25, 0.75, 0.79, None],
        'MSAVI': [0.25, 0.75, 0.79, None],
        'EVI':   [0.25, 0.75, 0.79, None],
        'EVI2':  [0.25, 0.75, 0.79, None],
        'NDWI':  [0.25, 0.75, 0.79, None],
        'NBR':   [0.25, 0.75, 0.79, None],
        'TCB':   [0.25, 0.75, 0.79, None],
        'TCG':   [0.25, 0.75, 0.79, None],
        'TCW':   [0.25, 0.75, 0.79, None]
    }
)
test_df = pd.DataFrame({
    'ID': [1]*13,
    "PixelID": ['1_1']*13,
    "Season": [2008]*13,
    'Type': ['Conifer']*13,
    "Density": [1434.54]*13,
    'SrvvR_1': [None]*13,
    'SrvvR_2': [90]*13,
    'SrvvR_3': [None]*13,
    'SrvvR_4': [90]*13,
    'SrvvR_5': [None]*13,
    'SrvvR_6': [90]*13,
    'SrvvR_7': [None]*13,
    'AssD_1': [None]*13,
    'AssD_2': ['2010-01-01']*13,
    'AssD_3': [None]*13,
    'AssD_4': ['2012-07-01']*13,
    'AssD_5': [None]*13,
    'AssD_6': ['2014-12-31']*13,
    'AssD_7': [None]*13,
    'Year': [2009, 2009, 2010, 2010, 2011, 2012, 2012, 2012, 2012, 2014, 2014, 2015, 2015],
    'DOY': [50, 360, 3, 83, 183, 34, 170, 190, 318, 298, 361, 10, 19],
    'NDVI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'SAVI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'MSAVI': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'EVI':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'EVI2':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'NDWI':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'NBR':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'TCB':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'TCG':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34],
    'TCW':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.84, 0.74, 0.34]
})

def test_mean_vi():
    assert match_vi(pivoted_df, test_df, 16).equals(exp_match_df)

def test_target_to_bin():
    '''
    Test for if target_to_bin() function correctly converts the target column into binary classes according to provided threshold.
    '''
    df_80 = target_to_bin(df_bin, 80)
    assert df_80['target'] == ['High', 'High', 'High', 'Low', 'Low', 'Low']
    
    # Test if Value Error is raised if threshold is not provided.
    with pytest.raises(ValueError, match=r"Threshold for classifying survival rate required*"):
        target_to_bin(df_bin)



