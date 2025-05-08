def target_to_bin(df, threshold):
    '''
    Map target to binary class "High"/"Low" survival rate base on given threshold.
    '''
    df['target'] = (df['target'] < threshold).map(
        {True: 'Low', False: 'High'})
    return df

