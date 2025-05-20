import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.pivot_data import pivot_df, match_vi, target_to_bin

# Test data for pivot_df()
test_pivot_df = pd.DataFrame({
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
    'AsssD_1': [None, '2009-07-01', '2009-07-01'],
    'AsssD_2': ['2010-07-01']*3,
    'AsssD_3': [None, None, None],
    'AsssD_4': ['2012-07-01']*3,
    'AsssD_5': [None, '2013-07-01', '2013-07-01'],
    'AsssD_6': ['2014-07-01', None, None],
    'AsssD_7': ['2015-07-01', None, None],
    'ImgDate': ['2001-01-01', '2002-01-01', '2003-01-01'],
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

# Expected Columns after pivoting
exp_pivot_cols = ['ID', 'PixelID', 'Density', 'Type',
           'Season', 'Age', 'target', 'SrvvR_Date']

# Expected values in 'SrvvR_Date' after pivoting
exp_pivot_date = ['2009-07-01', '2010-07-01', '2010-07-01',
                  '2012-07-01', '2012-07-01', '2013-07-01', '2014-07-01']

# Test data for match_vi()
test_match_df = pd.DataFrame({
    'ID': [1]*8 + [2]*8,
    "PixelID": ['1_1']*8 + ['2_1']*4 + ['2_2']*4,
    "Season": [2008]*16,
    'Type': ['Conifer']*8 + ['Mixed']*8,
    "Density": [1423.5]*8 + [2124.3]*8,
    'SrvvR_1': [90]*16,
    'SrvvR_2': [90]*16,
    'SrvvR_3': [None]*16,
    'SrvvR_4': [90]*16,
    'SrvvR_5': [None]*16,
    'SrvvR_6': [90]*16,
    'SrvvR_7': [90]*16,
    'AsssD_1': [None]*16,
    'AsssD_2': ['2010-01-01']*16,
    'AsssD_3': [None]*16,
    'AsssD_4': ['2012-07-01']*16,
    'AsssD_5': [None]*16,
    'AsssD_6': ['2014-12-31']*16,
    'AsssD_7': ['2015-12-31']*16,
    'ImgDate': ['2009-12-15', '2009-12-16', '2010-01-17', '2010-01-18', '2014-11-28',
                '2014-12-31', '2015-01-16', '2015-02-15']*2,
    'NDVI':  [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'SAVI':  [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'MSAVI': [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'EVI':   [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'EVI2':  [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'NDWI':  [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'NBR':   [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'TCB':   [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'TCG':   [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2,
    'TCW':   [0.1, 0.2, 0.3, 0.4, 0.85, 0.84, 0.74, 0.34]*2
})

# Pivoted dataframe
pivoted_df = pd.DataFrame(
    {
        'ID': [1]*4 + [2]*8,
        'PixelID': ['1_1']*4 + ['2_1']*4 + ['2_2']*4,
        'Density': [1434.54]*4 + [2124.3]*8,
        'Type':  ['Conifer']*4 + ['Mixed']*8,
        'Season': [2008]*12,
        'Age': [2, 4, 6, 7]*3,
        'target': [90.0]*12,
        'SrvvR_Date': ['2010-01-01', '2012-07-01', '2014-12-31', '2015-12-31']*3,
    }
)
# Spectral index columns
cols = ['NDVI', 'SAVI', 'MSAVI', 'EVI',
        'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG', 'TCW']

# Expected average signals
exp_avg = pd.concat([pd.Series([0.25, None, 0.79, None, 0.25, None, None, None, None, None, 0.79, None])]*10, axis=1)
exp_avg.columns = cols

# Test Data for target_to_bin()
df_bin = pd.DataFrame(
    {
        'ID': [1, 1, 2, 3, 4, 5],
        'PixelID': ['123_321', '123_321', '123_321', '123_321', '123_321', '123_321'],
        'Density': [1434.54, 1235, 3415.43, 3413.15, 2458.43, 13463],
        'Type':  ['Conifer', 'Conifer', 'Mixed', 'Mixed', 'Decidous', 'Decidous'],
        'Season': ['2025', '2025', '2025', '2025', '2024', '2015'],
        'Age': ['1', '2', '3', '4', '5', '6'],
        'target': [100, 95.6, 80.0, 74.6, 40.3, 0.0],
        'SrvvR_Date': ['2009-07-01', '2009-07-01', '2009-07-01', '2009-07-01', '2009-07-01', '2009-07-01']
    }
)
# Expected classified classes
exp_bin = [1, 1, 1, 0, 0, 0]




# 1. Tests for pivot_df()
def test_pivot_df_size():
    '''
    Test if pivot_df() function returns the correct number of rows and columns.
    '''
    pivoted_df = pivot_df(test_pivot_df)

    assert pivoted_df.columns.to_list() == exp_pivot_cols
    assert len(pivoted_df) == 7

def test_pivot_df_duplicates():
    '''
    Test if all duplicate rows are removed after pivoting.
    '''
    assert not pivot_df(test_pivot_df).duplicated().any()

def test_pivot_df_na():
    '''
    Test if all rows with missing values are removed after pivoting.
    '''
    assert not pivot_df(test_pivot_df).isna().any(axis=None)

def test_pivot_df_outlier():
    '''
    Test if outlier are removed after pivoting.
    '''
    assert (pivot_df(test_pivot_df)['Age'].isin([1, 2, 4, 5, 6])).all()

def test_pivot_df_merge():
    '''
    Test if the assessment dates are matched correctly to the survival records.
    '''
    pivoted_df = pivot_df(test_pivot_df).sort_values('Age')
    assert (pivoted_df['SrvvR_Date'] == exp_pivot_date).all()



# 2. Tests for match_vi()
def test_match_vi_rows():
    '''
    Test if all rows in the pivot dataframe is returned after matching.
    '''
    pivoted_df['SrvvR_Date'] = pivoted_df['SrvvR_Date'].astype(
        'datetime64[ns]').dt.date
    assert match_vi(pivoted_df, test_match_df, 16)[
        exp_pivot_cols].equals(pivoted_df)


def test_match_vi_cols():
    '''
    Test if the function returns the correct columns after matching.
    '''
    exp_match_cols = exp_pivot_cols + cols
    assert match_vi(pivoted_df, test_match_df,
                    16).columns.to_list() == exp_match_cols


def test_match_vi_values():
    '''
    Test if the average values are calculated correctly.
    '''
    assert match_vi(pivoted_df, test_match_df, 16)[cols].equals(exp_avg)
    
    

# 3. Tests for target_to_bin()
def test_target_to_bin():
    '''
    Test if the target column is correctly classified into binary classes.
    '''
    assert (target_to_bin(df_bin, 0.8)['target'] == exp_bin).all()


def test_target_to_bin_exception():
    '''
    Test if ValueError is raised if threshold is not given
    '''
    with pytest.raises(ValueError, match=r"Threshold for classifying survival rate required*"):
        target_to_bin(df_bin)
    with pytest.raises(ValueError, match=r"The threshold must be between 0 and 1*"):
        target_to_bin(df_bin, 80)
