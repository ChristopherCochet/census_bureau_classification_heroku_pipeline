# Script to train machine learning model.
import pandas as pd
import numpy as np
import dill as pickle
from model import (
    load_file,
    generate_feature_encoding,
    one_hot_encode_feature_df,
    train_model,
    get_best_model,
    print_summary,
    compute_model_metrics,
    inference,
    compute_slice_metrics,
    save_results,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import shuffle


# Model Training Script Main
############################
if __name__ == "__main__":

    # Data Loading and preprocessing
    ################################

    # define inputs
    train_file = "starter/data/census_clean.csv"
    model_dir = "starter/model/"

    # define variables
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    numeric_features = [
        "age",
        "fnlgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    target_var = "salary"

    # load data
    print("Loading data...")
    train_df = load_file(train_file)

    # shuffle, and reindex training data -- shuffling improves cross-validation accuracy
    print("Shuffling data...")
    train_df = shuffle(train_df)

    # get target df
    print("Retrieving labels...")
    target_df = train_df.pop(target_var)

    # encode categorical data and get final feature dfs
    print("Encoding data...")
    ct = generate_feature_encoding(
        train_df, cat_vars=cat_features, num_vars=numeric_features
    )

    feature_df = one_hot_encode_feature_df(train_df, ct)

    # Modelling
    ###########################################

    # initialize model list and dicts
    models = []
    roc_auc_dict = {}
    cv_std = {}
    res = {}

    # define number of processes to run in parallel
    num_procs = -1

    # shared model parameters
    verbose_lvl = 0

    # create models
    rf = RandomForestClassifier(
        n_estimators=150,
        n_jobs=num_procs,
        max_depth=25,
        min_samples_split=60,
        max_features=30,
        verbose=verbose_lvl,
    )

    gbc = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, loss="exponential", verbose=verbose_lvl
    )

    models.extend([rf, gbc])

    # parallel cross-validate models, using roc_auc as evaluation metric,
    # and print summaries
    print("Beginning cross validation...")
    for model in models:
        train_model(model, feature_df, target_df, num_procs, roc_auc_dict, cv_std)
        print_summary(model, roc_auc_dict, cv_std)

    # choose model with best auc_roc
    best_model = get_best_model(roc_auc_dict)

    # train best model on entire dataset
    print("Fit best performing model...")
    best_model.fit(feature_df, target_df)

    # best model metrics
    print("Compute metrics...")
    preds = inference(best_model, feature_df)
    precision, recall, fbeta = compute_model_metrics(target_df.values, preds)
    print(
        "Model metrics: precision = {}, recall = {}, fbeta(1) = {}".format(
            precision, recall, fbeta
        )
    )

    # compute slice metrics for education
    print("Compute slice metrics...")
    cat_feature = "education"
    cat_metric_df = compute_slice_metrics(
        model_dir, best_model, feature_df, target_df, cat_feature
    )
    print(cat_metric_df)

    # Save feature importances
    print("Save model and feature importances...")
    importances = best_model.feature_importances_
    feature_importances = pd.DataFrame(
        {"feature": feature_df.columns, "importance": importances}
    )
    feature_importances.sort_values(by="importance", ascending=False, inplace=True)
    feature_importances.set_index("feature", inplace=True, drop=True)

    # save results and model
    save_results(best_model, ct, roc_auc_dict[model], feature_importances, model_dir)
