from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GroupKFold


def build_rf_pipline(
    feat_select: str = None,
    drop_features: list = None,
    step_RFE: int = 1,
    num_feats_RFE: int = 4,
    min_num_feats_RFECV: int = 4,
    num_folds_RFECV: int = 5,
    scoring_RFECV: str = 'f1',
    random_state: int = 591,
    **kwargs
):

    if feat_select not in (None, 'RFE', 'RFECV'):
        raise ValueError('feat_select must be one of: {None, \'RFE\', \'RFECV\'}')

    if drop_features is not None and not isinstance(drop_features, list):
        raise ValueError('drop_features must be a list or None')

    drop_cols = (
        ['ID', 'PixelID'] if drop_features == None else ['ID', 'PixelID'] + drop_features
        )
    categorical_cols = ['Type']

    random_forest = RandomForestClassifier(random_state=random_state, **kwargs)

    preprocessor = make_column_transformer(
            ('drop',drop_cols),
            (OneHotEncoder(),categorical_cols),
            remainder='passthrough'
        )

    if feat_select not in (None, 'RFE', 'RFECV'):
        raise ValueError(
            'feat_select must be one of: {None, \'RFE\', \'RFECV\'}'
            )

    if feat_select == None:
        model_pipeline = make_pipeline(
            preprocessor,
            random_forest
        )

    elif feat_select == 'RFE':
        model_pipeline = make_pipeline(
            preprocessor,
            RFE(
                estimator=random_forest,
                n_features_to_select=num_feats_RFE,
                step=step_RFE,
            )
        )

    else:
        model_pipeline = make_pipeline(
            preprocessor,
            RFECV(
                estimator=random_forest,
                min_features_to_select=min_num_feats_RFECV,
                scoring=scoring_RFECV,
                n_jobs=-1,
                cv=GroupKFold(n_splits=num_folds_RFECV,
                              shuffle=True,
                              random_state=random_state),
            ),
        )

    return model_pipeline
