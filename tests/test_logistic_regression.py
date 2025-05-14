import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline

from src.model.logistic_regression import build_logreg_pipeline
from src.model.training_utils_losistic_regression import train_logreg_pipeline

# ------------------------------------------------------------------ #
# Fixtures                                                           #
# ------------------------------------------------------------------ #
@pytest.fixture()
def fake_df():
    """
    Create sample DataFrame for logistic regression tests using fixed values.
    """
    df = pd.DataFrame({
        'ID':       np.arange(1, 16),
        'PixelID':  np.arange(101, 116),
        'Type':     ['Decidous']*5 + ['Mixed']*5 + ['Conifer']*5,
        'NDVI':     [0.8, 0.6, 0.2, -0.5, -0.1, 0.3, 0.3, 0.6, 0.7, 0.8, 0.6, 0.2, -0.5, -0.1, 0.3],
        'SAVI':     [0.7, -0.9, 0.5, -0.5, 0.2, 0.1, 0.1, 0.5, 0.9, 0.9, -0.6, 0.2, -0.8, 0.3, 0.7],
        'MSAVI':    [0.4, 0.9, 0.6, 0.1, -0.8, -0.2, -0.9, 0.5, -0.5, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
        'EVI':      [0.5, 0.4, -0.3, 0.7, 0.9, 0.0, 0.3, -0.8, -0.2, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
        'EVI2':     [0.3, -0.4, 0.8, -1.0, 0.2, -0.5, 0.9, -0.6, 0.2, 0.7, -0.9, 0.5, -0.5, 0.2, 0.1],
        'NDWI':     [-0.7, 0.2, 0.6, 0.3, -0.2, 0.1, 0.1, 0.1, 0.5, -0.5, 0.3, 0.5, -0.9, 0.0, 0.7],
        'NBR':      [0.9, -0.6, 0.2, -0.8, 0.3, 0.1, -0.2, 0.4, 0.5, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],
        'TCB':      [-0.5, 0.3, 0.5, -0.9, 0.0, 0.7, 0.3, -0.4, 0.8, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],
        'TCG':      [0.6, -0.2, 0.4, 0.5, -0.3, 0.2, -0.2, 0.1, 0.1, 0.3, -0.8, -0.2, 2.5, 0.6, -0.4],
        'TCW':      [0.3, -0.8, -0.2, 2.5, 0.6, -0.4, 0.9, 0.6, 0.1, -0.2, 2.5, 0.6, -0.4, 0.9, 0.6],
        'Density':  [800]*15,
        'target':   [0]*8 + [1]*7
    })
    return df


# ------------------------------------------------------------------ #
# Basic construction & error handling                                #
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("fs", [None, "RFE", "RFECV"])
def test_is_pipeline(fs):
    pipe = build_logreg_pipeline(feat_select=fs)
    assert isinstance(pipe, Pipeline)


def test_invalid_feat_select():
    with pytest.raises(ValueError):
        build_logreg_pipeline(feat_select="banana")


# ------------------------------------------------------------------ #
# Fit end-to-end                                                     #
# ------------------------------------------------------------------ #
def test_pipeline_fits(fake_df):
    pipe = build_logreg_pipeline()
    trained, metrics = train_logreg_pipeline(pipe, fake_df, prop_train=0.7)
    check_is_fitted(trained)
    assert 0.0 <= metrics["accuracy"] <= 1.0