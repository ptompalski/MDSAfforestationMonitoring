from typing import List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
import os
import click
import joblib
import json
from sklearn.metrics import make_scorer,f1_score

def build_logreg_pipeline(
    feat_select: Optional[str] = None,                 # {None,'RFE','RFECV'}
    drop_features: Optional[List[str]] = None,
    step_RFE: int = 1,
    num_feats_RFE: int = 6,
    min_num_feats_RFECV: int = 6,
    num_folds_RFECV: int = 5,
    scoring_RFECV: str = "f1",
    random_state: int = 591,
    **logreg_kwargs,
):
    """
    Build a pipeline for logistic regression on the afforestation dataset.

    Parameters
    ----------
    feat_select : {None, 'RFE', 'RFECV'}, default=None
        Optional feature selection strategy: None, recursive feature elimination ('RFE'),
        or recursive feature elimination with cross-validation ('RFECV').
    drop_features : list of str or None, default=None
        Additional columns to drop.
    step_RFE : int, default=1
        Step size for RFE elimination.
    num_feats_RFE : int, default=6
        Number of features to select with RFE.
    min_num_feats_RFECV : int, default=6
        Minimum number of features for RFECV.
    num_folds_RFECV : int, default=5
        Number of folds for RFECV.
    scoring_RFECV : str, default='f1'
        Scoring metric for RFECV.
    random_state : int, default=591
        Random seed for reproducibility.
    **logreg_kwargs : dict
        Additional keyword arguments passed to `LogisticRegression`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An sklearn Pipeline object that applies preprocessing, optional feature selection,
        and logistic regression.
    """

    if feat_select not in (None, "RFE", "RFECV"):
        raise ValueError("feat_select must be one of {None, 'RFE', 'RFECV'}")
    if drop_features is not None and not isinstance(drop_features, list):
        raise ValueError("drop_features must be a list or None")

    # ------------------------------------------------------------------ #
    # Pre-processing: drop ID columns + optional extra, scale numerics,   #
    # one-hot 'Type'. The numeric list is the same as in the GBM tests.   #
    # ------------------------------------------------------------------ #
    drop_cols = (
        ['ID', 'PixelID', 'Season','SrvvR_Date'] if drop_features is None 
        else ['ID', 'PixelID', 'Season', 'SrvvR_Date'] + drop_features
    )

    numeric_cols = [
        "NDVI", "SAVI", "MSAVI", "EVI", "EVI2", "NDWI", "NBR",
        "TCB", "TCG", "TCW", "Density"
    ]
    categorical_cols = ["Type"]

    preprocessor = make_column_transformer(
        ("drop", drop_cols),
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        remainder="passthrough",
        force_int_remainder_cols=False,
    )

    base_clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
        **logreg_kwargs,
    )

    # ------------------------------------------------------------------ #
    # Assemble pipeline with optional RFE / RFECV                        #
    # ------------------------------------------------------------------ #
    if feat_select is None:
        return make_pipeline(preprocessor, base_clf)

    if feat_select == "RFE":
        selector = RFE(
            estimator=base_clf,
            n_features_to_select=num_feats_RFE,
            step=step_RFE,
        )
    else:  # RFECV
        selector = RFECV(
            estimator=base_clf,
            min_features_to_select=min_num_feats_RFECV,
            scoring=scoring_RFECV,
            n_jobs=-1,
            cv=GroupKFold(
                n_splits=num_folds_RFECV,
            ),
        )

    return make_pipeline(preprocessor, selector, base_clf)


@click.command()
@click.option('--feat_select', type=click.Choice(['None', 'RFE', 'RFECV']), default='None',
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
                help='Additional hyperparameters for LogisticRegression as JSON string')
def main(feat_select, drop_features, step_rfe, num_feats_rfe,
         min_num_feats_rfecv, num_folds_rfecv, scoring_rfecv,
         output_dir, random_state, kwargs_json):
    kwargs = json.loads(kwargs_json)
    
    feat_select = None if feat_select == 'None' else feat_select
    scoring_rfecv = make_scorer(f1_score,pos_label=0) if scoring_rfecv == 'f1' else scoring_rfecv
    
    
    pipeline = build_logreg_pipeline(
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

    if feat_select == None: model_name = "logistic_regression.joblib"
    elif feat_select == 'RFE': model_name = "logistic_regression_rfe.joblib"
    else: model_name = "logistic_regression_rfecv.joblib"
 
 
    model_path = os.path.join(output_dir, model_name)

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()