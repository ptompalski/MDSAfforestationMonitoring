import pytest
import numpy as np
import pandas as pd
from ..src.data.preprocess_features import target_to_bin

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
    pass

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