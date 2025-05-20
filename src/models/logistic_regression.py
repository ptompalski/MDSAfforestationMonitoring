"""
Pipeline builder for Logistic Regression on the afforestation data.

Features
--------
* Column dropping + One-Hot encoding (same as GBM pipeline)
* Optional feature-selection with RFE or RFECV
"""

from typing import List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold


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
    Construct a Scikit-Learn pipeline for Logistic Regression
    with the same signature as `build_gbm_pipeline`.

    Parameters
    ----------
    feat_select   : {None,'RFE','RFECV'}
    drop_features : list[str] | None
    step_RFE      : int      – step size for RFE elimination
    num_feats_RFE : int
    min_num_feats_RFECV : int
    num_folds_RFECV     : int
    scoring_RFECV       : str  – e.g. 'f1'
    random_state        : int
    **logreg_kwargs     : any extra kwargs -> LogisticRegression

    Returns
    -------
    sklearn.pipeline.Pipeline
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
        ['ID', 'PixelID', 'Season','SrvvR_Date' ] if drop_features is None 
        else ['ID', 'PixelID', 'Season', 'SrvvR_Date' ] + drop_features
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