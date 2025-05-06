import pyreadr
import pandas as pd
import numpy as np

def load_data(file):
    result = pyreadr.read_r(file)
    
    return result[None]
