# Matching Survey Records with VIs Signals

# Option 1: Keep records with survival rate measurements but no satellite data.
def mean_vi(df_melt, df, day_range, cols):
    '''
    Calculate mean VI signal: average of ± 16 days of Assessment Date.
    '''
    doy = df_melt['SrvvR_Date'].timetuple().tm_yday
    try:
        df = df.loc[[(df_melt['PixelID'], df_melt['SrvvR_Date'].year)]]
        df = df[df['DOY'].between(
            doy - day_range, doy + day_range, inclusive='both')]
        return df[cols].mean()
    except KeyError:
        return pd.Series([None] * len(cols), index=cols)


def match_vi(df_melt, df, day_range, cols):
    '''
    Match mean vi signal to pixel by Assessment Date.
    '''
    df['Key'] = list(zip(df['PixelID'], df['Year']))
    df.set_index('Key', inplace=True)
    df = df[cols + 'DOY']
    df_melt[cols] = None
    df_melt[cols] = df_melt.apply(
        lambda x: mean_vi(x, df, day_range, cols), axis=1)
    return df_melt


# Option 2: Removes records with no satellite data.
def mean_vi(df_melt, df, day_range, cols):
    '''
    Calculate mean VI signal: average of ± 16 days of Assessment Date.
    '''
    df = df.loc[[df_melt['key']]]
    df = df[df['DOY'].between(
        df_melt['doy'] - day_range, df_melt['doy'] + day_range, inclusive='both')]
    return df[cols].mean()


def match_vi(df_melt, df, day_range, cols):
    '''
    Match mean vi signal to pixel by Assessment Date.
    '''
    df['Key'] = list(zip(df['PixelID'], df['Year']))
    df.set_index('Key', inplace=True)
    df_melt['doy'] = df_melt['SrvvR_Date'].dt.day_of_year
    df_melt['key'] = list(
        zip(df_melt['PixelID'], df_melt['SrvvR_Date'].dt.year))
    df_melt = df_melt[df_melt['key'].isin(df.index)]
    df_melt[cols] = df_melt.apply(
        lambda x: mean_vi(x, df, day_range, cols), axis=1)
    return df_melt
