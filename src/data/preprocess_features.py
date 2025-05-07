import pandas as pd
import numpy as np
import click

def classify_species(x):
    """
    Classify site species type as 'Deciduous', 'Conifer', or 'Mixed' based on hardwood (HW) and softwood (SW) percentages.

    A site is classified as:
        - "Deciduous" if >= 80% of species are hardwood (HW),
        - "Conifer" if >= 80% are softwood (SW),
        - "Mixed" otherwise.

    Parameters
    ----------
        x : str
            A string containing HW:SW composition, formatted as 'HW <int>, SW <int>'.
    
    Returns
    ----------
    str
        one of "Decidous", "Conifer" or "Mixed"
    """
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
    """
    This function performs the following data cleaning. 
    1. Drop records of replanted sites
    2. Drop rows with missing spectral indices.
    3. Drop rows where the image year ('Year') precedes the planting year ('Season').
    4. Drop rows with invalid survival rates (SrvvR_1, ..., SrvvR_7 not in [0, 100]).
    5. Drop rows with invalid spectral index values (-1 to 1).
    6. Drop unnecessary columns (NmbrPlO, NmbrPlR, NmbrPlT, 'prevUse', 'SpcsCmp').

    Parameters
    ----------
        df : pd.DataFrame
            Original dataframe to be cleaned
            
    Returns
    ----------
        pd.DataFrame
            Cleaned data.
    """
    # Drop replanted sites
    df = df[(df['NmbrPlR'].isna()) | (df['NmbrPlR'] == 0)]
    # Drop rows with missing spectral indices
    df = df.dropna(subset=['NDVI', 'SAVI', 'MSAVI', 'EVI',
                        'EVI2', 'NDWI', 'NBR', 'TCB', 'TCG', 'TCW'])

    # remove records before planting
    df = df[df['Year']>= df['Season']]

    # drop out-of-range survival rates
    for i in ['SrvvR_1', 'SrvvR_2', 'SrvvR_3', 'SrvvR_4', 'SrvvR_5', 'SrvvR_6', 'SrvvR_7']:
        df = df[(df[i].between(0, 100, inclusive='both')) | df[i].isna()]

    # drop out-of-range indices
    for i in ['NDVI', 'SAVI', 'MSAVI', 'EVI', 'EVI2', 'NDWI', 'NBR']:
        df = df[df[i].between(-1, 1, inclusive='both')]
    
    # Impute type column
    df.loc[df['NmbrPlR'].isna(
    ), 'Type'] = df.loc[df['NmbrPlR'].isna(), 'SpcsCmp'].apply(classify_species)
    
    # Drop unnecessary columns
    df = df.drop(['NmbrPlO', 'NmbrPlR', 'NmbrPlT', 'prevUse', 'SpcsCmp'], axis=1)

    return df


def main():
    '''
    Command-line interface for preprocessing features of data.
    Preprocessing target will be performed in following steps. 
    '''
    pass

if __name__ == '__main__':
    main()