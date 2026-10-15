"""model.preprocess: the feature pipeline every model is built on."""

import numpy as np

from model.preprocess import SMOKING_STATUSES, WORK_TYPES, build_feature_pipeline


def test_feature_pipeline_outputs_14_columns_and_no_missing_values(dataset):
    X, y = dataset
    features = build_feature_pipeline().fit_transform(X, y)
    assert features.shape[1] == 14
    assert not np.isnan(features).any()


def test_feature_width_is_stable_when_the_training_data_lacks_categories(dataset):
    """The one-hot categories are pinned, so a pipeline fitted on a small slice of the data
    that is missing some categories still outputs 14 columns instead of fewer."""
    X, _ = dataset
    subset = X[X.work_type.isin(["Private", "children"])]
    pipe = build_feature_pipeline().fit(subset)
    assert pipe.transform(X.head(5)).shape[1] == 14


def test_category_lists_cover_every_value_in_the_real_data(dataset):
    X, _ = dataset
    assert set(X["work_type"].unique()) <= set(WORK_TYPES)
    cleaned_smoking = X["smoking_status"].replace("Unknown", "never smoked")
    assert set(cleaned_smoking.unique()) <= set(SMOKING_STATUSES)
