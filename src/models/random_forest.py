from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GroupKFold
import click
import json
import joblib
import os


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
    """
    Constructs a Scikit-Learn pipeline for training a Random Forest model 
    on afforestation data, with optional feature selection and preprocessing.

    Parameters
    ----------
    feat_select : {'RFE', 'RFECV', None}, default=None
        Feature selection method to apply:
        - 'RFE': Recursive Feature Elimination.
        - 'RFECV': Recursive Feature Elimination with Cross-Validation.
        - None: No feature selection is performed.

    drop_features : list of str, optional
        Additional features to drop from the dataset (e.g., indices dropped in prior selection steps). 
        Always drops 'ID', 'PixelID' 'Season', and 'SrvvR_Date' by default.

    step_RFE : int, default=1
        Number of features to remove at each iteration of the RFE process.

    num_feats_RFE : int, default=4
        Number of features to retain when using RFE.

    min_num_feats_RFECV : int, default=4
        Minimum number of features to retain when using RFECV.

    num_folds_RFECV : int, default=5
        Number of cross-validation folds to use during RFECV.

    scoring_RFECV : str, default='f1'
        Scoring metric used during RFECV. Must be a valid scoring string recognized by scikit-learn.
        See: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    random_state : int, default=591
        Random state seed for reproducibility.

    **kwargs : dict
        Additional keyword arguments passed to the `RandomForestClassifier` (e.g., hyperparameters like 
        `n_estimators`, `max_depth`, etc.).

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline consisting of preprocessing (column dropping, one-hot encoding) 
        and the Random Forest Classifier, optionally wrapped in RFE or RFECV.

    Raises
    ------
    ValueError
        If `feat_select` is not one of {None, 'RFE', 'RFECV'}.
        If `drop_features` is not None or a list.
    """
    if feat_select not in (None, 'RFE', 'RFECV'):
        raise ValueError('feat_select must be one of: {None, \'RFE\', \'RFECV\'}')

    if drop_features is not None and not isinstance(drop_features, list):
        raise ValueError('drop_features must be a list or None')

    drop_cols = (
        ['ID', 'PixelID', 'Season', 'SrvvR_Date'] if drop_features == None 
        else ['ID', 'PixelID', 'Season','SrvvR_Date'] + drop_features
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


@click.command()
@click.option('--feat_select', type=click.Choice([None, 'RFE', 'RFECV']), default=None,
                help='Feature selection method to apply')
@click.option('--drop_features', type=list, default=None,
                help='Comma-separated list of features to drop (e.g., "feat1,feat2")')
@click.option('--step_rfe', type=int, default=1,
                help='Number of features to remove at each iteration of RFE')
@click.option('--num_feats_rfe', type=int, default=4,
                help='Number of features to retain when using RFE')
@click.option('--min_num_feats_rfecv', type=int, default=4,
                help='Minimum number of features to retain when using RFECV')
@click.option('--num_folds_rfecv', type=int, default=5,
                help='Number of cross-validation folds to use during RFECV')
@click.option('--scoring_rfecv', type=str, default='f1',
                help='Scoring metric used during RFECV')
@click.option('--random_state', type=int, default=591,
                help='Random state seed for reproducibility')
@click.option('--output_dir', type=click.Path(file_okay=False), required=True,
              help='Directory to save pipeline model')
@click.option('--kwargs_json', type=str, default='{}',
                help='Additional hyperparameters for RandomForestClassifier as JSON string')
def main(feat_select, drop_features, step_rfe, num_feats_rfe,
         min_num_feats_rfecv, num_folds_rfecv, scoring_rfecv,
         output_dir, random_state, kwargs_json):
    """
    Command-line interface for building a Random Forest pipeline.
    """

    kwargs = json.loads(kwargs_json)

    pipeline = build_rf_pipline(
        feat_select=feat_select,
        drop_features=drop_features,
        step_RFE=step_rfe,
        num_feats_RFE=num_feats_rfe,
        min_num_feats_RFECV=min_num_feats_rfecv,
        num_folds_RFECV=num_folds_rfecv,
        scoring_RFECV=scoring_rfecv,
        random_state=random_state,
        **kwargs
    )

    model_name = "rf_model.joblib"
    model_path = os.path.join(output_dir, model_name)

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()