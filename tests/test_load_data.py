import pytest
import pyreadr
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.load_data import load_data

MOCK_DATA_RDS_PATH = "tests/test.rds"

mock_data = pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 1.2, -0.5, -1.1, 0.3],   # 1.2 and -1.1 are outliers
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],   
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [1.0, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -1.0, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})


@pytest.fixture
def setup_mock_data():
    """Fixture create test rds file."""
    pyreadr.write_rds(MOCK_DATA_RDS_PATH, mock_data)
    return MOCK_DATA_RDS_PATH


def test_file_exists(setup_mock_data):
    """Test if the file exists."""
    assert os.path.exists(setup_mock_data)


def test_load_data_success(setup_mock_data):
    """Test the raised exception when the dataframe is empty."""
    df = load_data(setup_mock_data)
    assert type(df) is pd.DataFrame


def test_columns(setup_mock_data):
    """Test the shape of the loaded dataframe."""
    df = load_data(setup_mock_data)
    assert df.columns.tolist() == mock_data.columns.tolist()




