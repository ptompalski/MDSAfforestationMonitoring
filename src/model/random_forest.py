from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.data_split import data_split
import click
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GroupKFold


def build_rf_pipline(    
    feat_select: str = None,
    drop_features: list = [],
    num_feats_RFE: int = 4,
    min_num_feats_RFECV: int = 4,
    num_folds: int = 5,
    **kwargs
):

    drop_cols = ['ID', 'PixelID'] + drop_features   # ID cols ignored, option to drop others
    categorical_cols = ['Type']
    
    preprocessor = make_column_transformer(
            ('drop',drop_cols),
            (OneHotEncoder(),categorical_cols),
            remainder='passthrough'
        )                  # One-Hot Encode type columns  


    if feat_select not in (None, 'RFE', 'RFECV'):
        raise ValueError(
            'feat_select must be one of: {None, \'RFE\', \'RFECV\'}'
            )

    # no feature selection
    if feat_select == None:
        model_pipeline = make_pipeline(
            preprocessor,
            RandomForestClassifier(**kwargs)
        )

    # feature selection with RFE
    elif feat_select == 'RFE':
        model_pipeline = make_pipeline(
            preprocessor,
            RFE(
                estimator=RandomForestClassifier(**kwargs),
                n_features_to_select=num_feats_RFE
            ),
            RandomForestClassifier(**kwargs)
        )

    # Feature selection with RFECV
    else:
        model_pipeline = make_pipeline(
            preprocessor,
            RFECV(
                estimator=RandomForestClassifier(**kwargs),
                min_features_to_select=min_num_feats_RFECV,
                scoring='f1',                                    # use f1 score to handle imbalance
                n_jobs=-1,                                       # parallelize cross-validation if possible
                cv=GroupKFold(n_splits=num_folds)
            ),
            RandomForestClassifier(**kwargs)
        )

    return model_pipeline


def train_and_test_model(X_train, y_train, X_test, y_test, pipeline):

    pipeline.fit(X_train, y_train, groups=y_train['ID'])
    score = pipeline.score(X_test, y_test)
    return score

@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help='Path to train data. Expecting parquet format.')
def main(input_path):
    df = pd.read_parquet(input_path)

    df_train, df_test = data_split(df)
    X_train, X_test = df_train.drop(columns=['target']), df_train['target']
    y_train, y_test = df_test.drop(columns=['target']), df_test['target']

    pipeline = build_rf_pipline()
    score = train_and_test_model(X_train, y_train, X_test, y_test, pipeline)
    print(f'Model score: {score}')