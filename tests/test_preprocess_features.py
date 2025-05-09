import pytest
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.preprocess_features import classify_species, data_cleaning, create_density_feature

# expected columns of dataframe after cleaning
expected_columns = [
    'ID', 'PixelID', 'Area_ha', 'Season', 'Planted', 'Type', 'SrvvR_1',
    'SrvvR_2', 'SrvvR_3', 'SrvvR_4', 'SrvvR_5', 'SrvvR_6', 'SrvvR_7',
    'AssD_1', 'AssD_2', 'AssD_3', 'AssD_4', 'AssD_5', 'AssD_6', 'AssD_7',
    'Year', 'DOY', 'NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2',
    'NDWI', 'NBR', 'TCB', 'TCG', 'TCW']

feature_cols = [
    'NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2',
    'NDWI', 'NBR', 'TCB', 'TCG', 'TCW'
]

# baseline test dataframe with no outliers
df_no_outliers =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

# create test dataframes with outliers in each index
df_outlier_ndvi =  pd.DataFrame({
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

df_outlier_savi =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [2.7, -1.9, 0.5, -1.5, 0.2, 0.1],   # 2.7, -1.9, and -1.5 are outliers
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 1.0, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_msavi =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],   
    'MSAVI': [1.4, 0.9, 0.6, 0.1, -1.8, -0.2],  # 1.4 and -1.8 are outliers
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_evi =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],  
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [1.5, 0.4, -1.3, 0.7, 0.9, -2.1],   # 1.5, -1.3, and -2.1 are outliers 
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_evi2 =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [1.3, -1.4, 0.8, -1.0, -1.2, -0.5], # 1.3 and -1.2 are outliers 
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_ndwi =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-1.7, 0.2, 0.6, 1.3, -0.2, 0.1], # -1.7 and 1.3 are outliers
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_nbr =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -1.4, 0.8, -1.0, 1.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, -0.1],  
    'NBR':   [0.9, -0.6, 1.2, -1.8, 0.3, 0.1],   # 1.2 and -1.8 are outliers
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

# baseline test dataframe with no outliers in vegetation indices
df_no_outliers =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

# create test dataframes with outliers in each index
df_outlier_ndvi =  pd.DataFrame({
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

df_outlier_savi =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [2.7, -1.9, 0.5, -1.5, 0.2, 0.1],   # 2.7, -1.9, and -1.5 are outliers
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 1.0, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_msavi =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],   
    'MSAVI': [1.4, 0.9, 0.6, 0.1, -1.8, -0.2],  # 1.4 and -1.8 are outliers
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_evi =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],  
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [1.5, 0.4, -1.3, 0.7, 0.9, -2.1],   # 1.5, -1.3, and -2.1 are outliers 
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_evi2 =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [1.3, -1.4, 0.8, -1.0, -1.2, -0.5], # 1.3, -1.4, and -1.2 are outliers 
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_ndwi =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-1.7, 0.2, 0.6, 1.3, -0.2, 0.1], # -1.7 and 1.3 are outliers
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_outlier_nbr =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, -0.1],  
    'NBR':   [0.9, -0.6, 1.2, -1.8, 0.3, 0.1],   # 1.2 and -1.8 are outliers
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

# several tests cases for missing indices
df_missing_vis_1 = pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [pd.NA, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, pd.NA, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, -0.1],  
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})

df_missing_vis_2 = pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.1, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, 0.3, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, pd.NA, 0.1, -0.8, -0.2],  
    'EVI':   [pd.NA, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, pd.NA, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, pd.NA, -0.1],  
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})
 
df_missing_vis_3 = pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),'Year':np.arange(2001,2007),'DOY':[1]*6,
    
    'NDVI':  [0.1, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, 0.3, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.1, 0.1, -0.8, -0.2],  
    'EVI':   [0.1, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, 0.1, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, 0.1, -0.1],  
    'NBR':   [pd.NA, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, pd.NA, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, pd.NA, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, pd.NA, 0.6, -0.4],    
})

# test case with imaging years before planting
df_img_before_plant =  pd.DataFrame({
    'ID':np.arange(1,7),"PixelID":np.arange(101,107),"Area_ha": np.arange(100,700,100,dtype=float),
    "Season": np.arange(2001,2007,dtype=float),"PlantDt": [pd.NA]*6,"prevUse": ['AG']*6,
    'Planted': np.arange(1000,7000,1000,dtype=float),'SpcsCmp': ['HW 100, SW 0']*6,'Type': ['Decidous']*6,
    
    'SrvvR_1':[pd.NA]*6,'SrvvR_2':[pd.NA]*6,'SrvvR_3':[pd.NA]*6,'SrvvR_4':[pd.NA]*6,
    'SrvvR_5':[pd.NA]*6,'SrvvR_6':[pd.NA]*6,'SrvvR_7':[pd.NA]*6,
    
    'AssD_1':[pd.NA]*6,'AssD_2':[pd.NA]*6,'AssD_3':[pd.NA]*6,'AssD_4':[pd.NA]*6,
    'AssD_5':[pd.NA]*6,'AssD_6':[pd.NA]*6,'AssD_7': [pd.NA]*6,
    
    'NmbrPlO':[pd.NA]*6,'NmbrPlR':[pd.NA]*6,'NmbrPlT':[pd.NA]*6,
    'ImgDate':pd.date_range(start="2001-01-01", end="2007-01-01",freq='YE'),
    
    'Year':np.arange(1995,2011,3), # some years before planting
    
    'DOY':[1]*6, 
    
    'NDVI':  [0.8, 0.6, 0.2, -0.5, -0.1, 0.3],  
    'SAVI':  [0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
    'MSAVI': [0.4, 0.9, 0.6, 0.1, -0.8, -0.2],  
    'EVI':   [0.5, 0.4, -0.3, 0.7, 0.9, 0.1],    
    'EVI2':  [0.3, -0.4, 0.8, -1.0, 0.2, -0.5],  
    'NDWI':  [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1],   
    'NBR':   [0.9, -0.6, 0.2, -0.8, 0.3, 0.1],   
    'TCB':   [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7],   
    'TCG':   [0.6, -0.2, 0.4, 0.5, -0.3, 0.2],   
    'TCW':   [0.3, -0.8, -0.2, 0.5, 0.6, -0.4],    
})
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
        
def test_columns_dropped():
    '''
    Test to ensure correct columns are dropped after cleaning
    ''' 
    # expected columns after cleaning
    assert data_cleaning(df_no_outliers).columns.to_list() ==  expected_columns
 
@pytest.mark.parametrize(
    'input_df,expected_dropped_rows',
    [
        (df_outlier_ndvi,[2,4]),
        (df_outlier_savi,[0,1,3]),
        (df_outlier_msavi,[0,4]),
        (df_outlier_evi,[0,2,5]),
        (df_outlier_evi2,[0,1,4]),
        (df_outlier_ndwi,[0,3]),
        (df_outlier_nbr,[2,3] )        
    ]
)   
def test_outlier_indices(input_df,expected_dropped_rows):
    '''
    Test to ensure vegetation indices are within correct range after cleaning
    '''
    expected_output = input_df.drop(index=expected_dropped_rows)[expected_columns]
    assert data_cleaning(input_df).equals(expected_output)

@pytest.mark.parametrize(
    'input_df',
    [df_missing_vis_1,df_missing_vis_2,df_missing_vis_3]
)
def test_missingness(input_df):
    '''
    Test to ensure no missing values in features after cleaning.
    Missing values in survival rate are allowed at this stage.
    '''
    assert not data_cleaning(input_df)[feature_cols].isna().any().any()
    
def test_image_year_after_planting():
    '''
    Test to ensure only records with imaging after planting are kept.
    '''
    output = data_cleaning(df_img_before_plant)
    assert (output['Year'] >= output['Season']).all()

def test_create_density_feature():
    '''
    Test to ensure density feature is created correctly.
    '''
    output_with_density = create_density_feature(df_no_outliers)
    assert ('Planted', 'Area_ha') not in output_with_density.columns
    assert 'Density' in output_with_density.columns