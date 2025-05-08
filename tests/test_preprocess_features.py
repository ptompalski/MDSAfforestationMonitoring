import pytest
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.preprocess_features import classify_species,data_cleaning

#columns for creating test data
columns = [
    "ID", "PixelID", "Area_ha", "Season", "PlantDt", "prevUse", "Planted",
    "SpcsCmp", "Type"
] + [f"SrvvR_{i}" for i in range(1, 8)] + [f"AssD_{i}" for i in range(1, 8)] + [
    "NmbrPlO", "NmbrPlR", "NmbrPlT", "ImgDate", "Year", "DOY"
]


def test_classify_species():
    '''
    Test for correctness of classify_species() function
    '''
    # testing typical use-cases
    assert classify_species('HW 100, SW 0') == 'Decidous'
    assert classify_species('HW 80, SW 20') == 'Decidous'
    assert classify_species('HW 70, SW 30') == 'Mixed'
    assert classify_species('HW 50, SW 50') == 'Mixed'
    assert classify_species('HW 30, SW 70') == 'Mixed'
    assert classify_species('HW 20, SW 80') == 'Conifer'
    assert classify_species('HW 0, SW 100') == 'Conifer'
    
    # test erroneous use-cases
    with pytest.raises(ValueError):
        classify_species('an invalid string')
        
    with pytest.raises(ZeroDivisionError):
        classify_species('HW 0, SW 0')
        
def test_outlier_survival_rates():
    '''
    Test to ensure survival rates are between 0% and 100% after cleaning
    '''
    pass
    
def test_outlier_indices():
    '''
    Test to ensure vegetation indices are within correct range after cleaning
    '''
    pass
    
    
def test_missingness():
    '''
    Test to ensure no missing values in features after cleaning.
    Missing values in survival rate are allowed at this stage.
    '''
    pass

def test_image_year_after_planting():
    '''
    Test to ensure only records with imaging after planting are kept.
    '''
    pass