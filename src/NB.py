#####################################################################################################################################################################

## Naive Bias

# from psutil import STATUS_LOCKED
import os
from tokenize import Triple
from functions import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from functools import partial
import mlflow
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
import yaml
import os
from hyperopt import space_eval
from sklearn.naive_bayes import MultinomialNB

config_path = os.path.join('/home/asdf/prj/MLFlow/config/config.yaml')
config = yaml.safe_load(open(config_path))['train_model']

## Preprocessed data

test_preprocessed = pd.read_csv(config['data_path_test'], sep=';')

train_preprocessed = pd.read_csv(config['data_path_train'], sep=';')

# Splitting

train_X = train_preprocessed.review.array
train_y = train_preprocessed.sentiment.array

test_X = test_preprocessed.review.array
test_y = test_preprocessed.sentiment.array

class_names = ['negative', 'positive']

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment(experiment_name = 'Naive Bias')

# Train

params = {"random_state": config['random_state'], 'solver': config['solver']}
cv = StratifiedShuffleSplit(n_splits=config['n_splits'], test_size=config['test_size'], random_state=config['random_state'])

nb_pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer()),
                        ('clf', MultinomialNB()),
])

with mlflow.start_run(run_name=config["run_name"], tags = {config['tag_name_key']: config['tag_name_value']}):
    model_result(nb_pipeline,  train_X, train_y, train_X, train_y, model_name = 'Naive Bias')

    ROC_AUC_score_train = round(roc_auc_score(train_y, nb_pipeline.predict_proba(train_X)[:, 1]),2)
    mlflow.log_params(params)
    mlflow.log_metric("ROC_AUC_train_NB", ROC_AUC_score_train)

    # Test

    model_result(nb_pipeline,  train_X, train_y, test_X, test_y, model_name = 'Naive Bias')

    ROC_AUC_score_test = round(roc_auc_score(test_y, nb_pipeline.predict_proba(test_X)[:, 1]),2)
    mlflow.log_params(params)
    mlflow.log_metric("ROC_AUC_train_NB", ROC_AUC_score_test)

# Search

    search_space = {
        'clf__alpha': hp.choice('clf__penalty', [[0, 0.5, 1.0, 5, 10]]),
        'clf__fit_prior': hp.uniform('clf__fit_prior', [True, False])
    }

    trials = Trials()

    best = fmin(
            fn = objective,
            space = search_space,
            algo = tpe.suggest,
            max_evals = 2,
            trials = trials,
            rstate = np.random.default_rng(1),
            show_progressbar = True
    )
    print(trials.trials)

    best_params = space_eval(search_space, best)
    mlflow.log_params(best_params)


    nb_pipeline_tuned = Pipeline([('tfidf', TfidfVectorizer()),
                ('clf', MultinomialNB(alpha = 5, fit_prior = True)),
                ])

    model_result(nb_pipeline_tuned,  train_X, train_y, test_X, test_y, model_name = 'Naive Bias')

    ROC_AUC_score_nb_tuned = round(roc_auc_score(test_y, nb_pipeline_tuned.predict_proba(test_X)[:, 1]),2)

    r_a_score(nb_pipeline_tuned, train_X, train_y, test_X, test_y)

    # Plots
    # Zip coefficients and names together and make a DataFrame

    zipped = zip(grid_search.best_estimator_.get_params()['steps'][0][1].get_feature_names(), grid_search.best_estimator_.get_params()['steps'][1][1].coef_[0])
    df_NB = pd.DataFrame(zipped, columns=["feature", "value"])

    # Sort the features by the absolute value of their coefficient

    df_NB["abs_value"] = df_NB["value"].apply(lambda x: abs(x))
    df_NB = df_NB.sort_values(["abs_value"], ascending=False)
    df_NB["colors"] = df_NB["abs_value"].apply(lambda x: "#316879" if x < 7 else "#f47a60")


    max_green = df_NB.loc[df_NB.colors == '#f47a60'][:20]
    max_red = df_NB.loc[df_NB.colors == '#316879'][:20]
    fi = pd.concat([max_red, max_green]).sort_values(["abs_value"], ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    sns.set (style = "whitegrid")
    sns.barplot(x="value",
                y="feature",
                data=fi,
                color = '#f47a60'
                );
    ax.set_title("Top 20 Features", fontsize=25,  color='#4f4e4e');
    ax.set_xlabel("Coefficients", fontsize=20,  color='#4f4e4e');
    ax.set_ylabel("Feature names", fontsize=20,  color='#4f4e4e');
    sns.despine ();
    plt.xticks (size = 16,  color='#4f4e4e');
    plt.yticks (size = 16,  color='#4f4e4e');


