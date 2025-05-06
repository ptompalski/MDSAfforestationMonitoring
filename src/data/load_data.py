import pyreadr
import pandas as pd
import numpy as np

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
