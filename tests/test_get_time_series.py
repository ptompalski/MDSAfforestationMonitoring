import pytest
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.get_time_series import (
    _get_summary_statistics, _get_raw_sequence,
    split_interim_dataframe, process_and_save_sequences
)

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
    
