import pandas as pd


def target_to_bin(df, threshold):
    '''
    Map target to binary class "High"/"Low" survival rate base on given threshold.
    '''
    df['target'] = (df['target'] < threshold).map(
        {True: 'Low', False: 'High'})
    return df


def melt_df(df):
    '''
    Pivot data to combine Survival Rates and Assessment Dates into two separate columns. 
    '''
    df_sr = df.melt(
        id_vars=['ID', 'PixelID', 'Density', 'Type', 'Season'], value_vars=[1, 2, 3, 4, 5, 6, 7], 
        var_name='Age', 
        value_name='target'
    ).dropna(axis=0, subset='target').drop_duplicates()


    df_ad = df[['ID', 'PixelID','AsssD_1', 'AsssD_2', 'AsssD_3', 'AsssD_4', 'AsssD_5', 'AsssD_6', 'AsssD_7']]
    
    df_ad.columns = ['ID', 'PixelID', 1, 2, 3, 4, 5, 6, 7]
    df_ad = df_ad.melt(
        id_vars=['ID', 'PixelID'], 
        value_vars=[1, 2, 3, 4, 5, 6, 7], 
        var_name='Age', value_name='SrvvR_Date'
    ).dropna(axis=0, subset='SrvvR_Date').drop_duplicates()
    
    df = df_sr.merge(df_ad, on=['ID', 'PixelID', 'Age'])
    
    return df


