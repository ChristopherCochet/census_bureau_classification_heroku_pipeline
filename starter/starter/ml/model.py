# Helper Modeling Functions
###########################

import pandas as pd
import numpy as np
import dill as pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, precision_score, recall_score  # roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_file(file):
    """
    Loads csv to pd dataframe

    Inputs
    ------
    file : string
        The cleaned census csv dataset file

    Returns
    -------
    dataframe : pandas dataframe
    """
    return pd.read_csv(file)


def get_columns_from_transformer(column_transformer, input_colums):
    """
    Loads csv to pd dataframe

    Inputs
    ------
    column_transformer : ColumnTransformer
        The sklearn transformer used to process and encode features

    input_colums : list
        The list of feature column names

    Returns
    -------
    col_name : list
        The list of transformed column names
    """

    col_name = []

    for transformer_in_columns in column_transformer.transformers_:
        # the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names(raw_col_name)
        except AttributeError:
            # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names, np.ndarray):  # eg.
            col_name += names.tolist()
        elif isinstance(names, list):
            col_name += names
        elif isinstance(names, str):
            col_name.append(names)

    # print(col_name)
    return col_name


def generate_feature_encoding(df, cat_vars=None, num_vars=None):
    """performs one-hot encoding on all categorical variables and
       combines result with continuous variables
    """

    def df_to_numeric(df):
        import pandas

        return df.apply(pandas.to_numeric)

    ohe = OneHotEncoder()
    # can't pickle the following :(
    # numft = FunctionTransformer(lambda x: x.apply(pd.to_numeric), accept_sparse=True)
    numft = FunctionTransformer(df_to_numeric, accept_sparse=True)
    ct = ColumnTransformer(
        [("ohe", ohe, cat_vars), ("nuft", numft, num_vars)], remainder="drop"
    )

    ct.fit(df)
    return ct


def one_hot_encode_feature_df(df, col_transformer):
    """performs one-hot encoding on all categorical variables
       and combines result with continuous variables

    Inputs
    ------
    df : pandas dataframe
        The feature dataframe to be processed
    column_transformer : ColumnTransformer
        The sklearn transformer used to process and encode features

    Returns
    -------
    df : pandas dataframe
        The processed feature dataframe
    """

    processed_df = pd.DataFrame.sparse.from_spmatrix(col_transformer.transform(df))
    processed_df.columns = get_columns_from_transformer(col_transformer, df.columns)
    return processed_df


def get_target_df(df, target):
    """returns target dataframe"""
    return df[target]


def train_model(model, feature_df, target_df, num_procs, roc_auc_dict, cv_std):
    """performs cross validation with the model provided and stores
       performance in a dictionary

    Inputs
    ------
    model : ???
        Best performing trained machine learning model.
    df : pandas dataframe
        The preprocessed feature dataframe
    feature_df : pandas dataframe
        The preprocessed feature dataframe
    target_df : pandas dataframe
        The label dataframe
    num_procs : pint
        Number of processors used for the models cross validation training
    roc_auc_dict : dictionary
        Trained machine learning models and their roc_auc scores.
    cv_std : dictionary
        Trained machine learning models and their roc_auc scores std deviation.

    Returns
    -------

    """
    roc_auc = cross_val_score(
        model, feature_df, target_df, cv=5, n_jobs=num_procs, scoring="roc_auc_ovr"
    )
    roc_auc_dict[model] = roc_auc
    cv_std[model] = np.std(roc_auc)
    # import sklearn.metrics
    # sorted(sklearn.metrics.SCORERS.keys())


def get_best_model(roc_auc_dict):
    """return the machine learning model with the best performance

    Inputs
    ------
    roc_auc_dict : dictionary
        Trained machine learning models cross val roc_auc scores.

    Returns
    -------
    model : ???
        Best performing trained machine learning model.
    """

    best_score = 0
    for key, value in roc_auc_dict.items():
        if np.mean(value) >= best_score:
            model = key
    print("\nModel with highest roc_auc:")
    print(model)
    return model


def print_summary(model, roc_auc_dict, cv_std):
    """ Prints performance of the ml model provided

    Inputs
    ------
    model : ???
        Trained machine learning model.
    roc_auc_dict : dictionary
        Trained machine learning models cross val roc_auc scores.
    cv_std : dictionary
        Trained machine learning models cross val roc_auc scores std deviations.


    Returns
    -------

    """
    print("\nModel:\n", model)
    print("Average roc_auc :\n", roc_auc_dict[model])
    print("Standard deviation during Cross Validation:\n", cv_std[model])


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : pandas dataframe
        The label dataframe
    preds : np.array
        Predicted labels.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, pos_label=">50K", zero_division=1)
    precision = precision_score(y, preds, pos_label=">50K", zero_division=1)
    recall = recall_score(y, preds, pos_label=">50K", zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for predictions.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions


def compute_slice_metrics(model_dir, model, df, y, cat_feature):
    """ Function that computes performance on model slices

    Computes and saves the performance metrics when the value of a given feature
    is held fixed.
    E.g. for education, it would print out the model metrics for each slice of
    data that has a particular value for education. You should have one set of
    outputs for every single unique value in education.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    df : pandas dataframe
        The preprocessed feature dataframe
    y : pandas dataframe
        The label dataframe
    cat_feature : string
        The column name of the categorical feature used to slice the data

    Returns
    -------
    preds : pandas dataframe
        the model's performance metrics for each slice of the data
    """

    print("Compute metrics for slice {}...".format(cat_feature))
    slice_results_dict = {}
    cat_cols = [col for col in df.columns if cat_feature in col]

    for col in cat_cols:
        filt = df[col] == 1
        df_filt = df[filt]
        y_filt = y.iloc[df[filt].index]
        preds = inference(model, df_filt)
        precision, recall, fbeta = compute_model_metrics(y_filt.values, preds)
        slice_results_dict[col] = [precision, recall, fbeta]

    res_df = pd.DataFrame.from_dict(
        slice_results_dict, orient="index", columns=["precision", "recall", "fbeta"]
    )

    print("Saving metrics slice {}...".format(cat_feature))
    res_df.to_csv(model_dir + "slice_output.txt", index=True)

    return res_df


def save_results(
    model, feature_encoding_transformer, roc_auc_list, feature_importances, model_dir
):
    """ Saves model, model summary and feature importances

   Inputs
    ------
    model : ???
        Trained machine learning model
    roc_auc_list : numpy.ndarray
        The roc_auc cross validation score of the model
    feature_importances : pandas dataframe
        The importance score for each feature dataframe
    model_dir : string
        Path to the model directory where model related artifacts are saved

    Returns
    -------
    """

    # model
    encoding_file_name = "census_feature_encoding.pkl"
    pickle.dump(
        feature_encoding_transformer, open(model_dir + encoding_file_name, "wb")
    )

    # Model name
    with open(model_dir + "model.txt", "w") as file:
        file.write(str(model))

    # model
    model_file_name = "census_model.pkl"
    pickle.dump(model, open(model_dir + model_file_name, "wb"))

    # feature importance
    feature_importances.to_csv(model_dir + "feature_importances.csv")
