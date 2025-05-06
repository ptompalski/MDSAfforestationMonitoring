import pandas as pd
import numpy as np

def classify_species(x):
    hw = int(x.split(',')[0].strip('HW '))
    sw = int(x.split(',')[1].strip('SW '))
    sum = hw + sw
    if (hw/sum >= 0.8):
        return "Decidous"
    elif (sw/sum >= 0.8):
        return "Conifer"
    else:
        return "Mixed"


def data_cleaning(df):
    # Drop replanted sites
    df = df[(df['NmbrPlR'].isna()) | (df['NmbrPlR'] == 0)]
    # Drop rows with missing spectral indices
    df = df.dropna(subset=['NDVI', 'SAVI', 'MSAVI', 'EVI',
                        'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG', 'TCW'])

    # remove records before planting
    df = df[df['Year']>= df['Season']]

    # drop out-of-range survival rates
    for i in ['SrvvR_1', 'SrvvR_2', 'SrvvR_3', 'SrvvR_4', 'SrvvR_5', 'SrvvR_6', 'SrvvR_7']:
        df = df[(df[i] <= 100) | df[i].isna()]

    # drop out-of-range indices
    for i in ['NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR']:
        df_train = df_train[df_train[i].between(-1, 1, inclusive='both')]
    
    # Impute type column
    df.loc[df['NmbrPlR'].isna(
    ), 'Type'] = df.loc[df['NmbrPlR'].isna(), 'SpcsCmp'].apply(classify_species)

    return df

def data_split(df):
    id_list = np.arange(df['ID'].min(), df['ID'].max()+1)
    # initialize split parameters
    SEED = 591
    prop_train = 0.7
    n_training_sites = int(prop_train*len(id_list))
    np.random.seed(SEED)
    # randomly select proportion of sites for training set
    training_ids = np.random.choice(id_list, size=n_training_sites, replace=False)

    # filter the data for site IDs selected in the training set
    df_train = df[df['ID'].isin(training_ids)]
    df_test = df[~df['ID'].isin(training_ids)]
    
    return df_train, df_test