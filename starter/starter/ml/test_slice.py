import pandas as pd
import numpy as np
import dill as pickle
import pytest
from sklearn.compose import ColumnTransformer
from .train_model import one_hot_encode_feature_df, generate_feature_encoding


@pytest.fixture
def data():
    """ Retrieve Cleaned Dataset """
    train_file = "../../data/census_clean.csv"
    df = pd.read_csv(train_file)
    return df


def test_data_shape(data):
    """ Check that data has no null value """
    assert (
        data.shape == data.dropna().shape
    ), "Dropping null changes shape of dataframe."


def test_data_char_cleaned(data):
    """ Check that there are no ? characters in the categorical features """
    cat_col = data.select_dtypes(include=[object]).columns
    for col in cat_col:
        filt = data[col] == "?"
        assert filt.sum() == 0, f"Found ? character in feature {col}"


def test_data_column_name_cleaned(data):
    """ Check that there are no spaces in the column names """
    col_names = data.columns
    for col in col_names:
        assert " " not in col, f"Found space character in feature {col}"


def test_one_generate_feature_encoding(data):
    """ Check that the feature encoding column transformer object is created """
    num_vars = data.select_dtypes(include=np.number).columns
    cat_vars = data.select_dtypes(include=[object]).columns

    ct = generate_feature_encoding(data, cat_vars=cat_vars, num_vars=num_vars)

    assert isinstance(
        ct, ColumnTransformer
    ), "generate_feature_encoding returned wrong type!"


def test_one_hot_encode_feature(data):
    """ Check that the data is processed and encoded successfully using the column transformer pickle file """
    feature_file = "../../model/census_feature_encoding.pkl"

    with open(feature_file, "rb") as file:
        ct = pickle.load(file)

    res_df = one_hot_encode_feature_df(data, ct)
    assert isinstance(
        res_df, pd.DataFrame
    ), "one_hot_encode_feature_df returned wrong type!"


# def test_slice_averages(data):
#     """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
#     for cat_feat in data["categorical_feat"].unique():
#         avg_value = data[data["categorical_feat"] == cat_feat]["numeric_feat"].mean()
#         assert (
#             2.5 > avg_value > 1.5
#         ), f"For {cat_feat}, average of {avg_value} not between 2.5 and 3.5."
