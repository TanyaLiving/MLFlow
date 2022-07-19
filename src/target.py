

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
import hyperopt.pyll.stochastic

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
mlflow.set_experiment(experiment_name = config["experiment_name"])

## Logistic regression

# Baseline

params = {"random_state": config['random_state'], 'solver': config['solver']}

cv = StratifiedShuffleSplit(n_splits=config['n_splits'], test_size=config['test_size'], random_state=config['random_state'])

my_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', LogisticRegression(random_state = params['random_state'], solver = params['solver']))
])

# Train

with mlflow.start_run(run_name=config["run_name"], tags = {config['tag_name_key']: config['tag_name_value']}):
    model_result(my_pipeline,
                train_X, 
                train_y,
                train_X, 
                )
                
    ROC_AUC_train_LR = round(roc_auc_score(train_y, my_pipeline.predict_proba(train_X)[:, 1]), 2)
    
    mlflow.log_params(params)
    mlflow.log_metric("ROC_AUC_train_LR", ROC_AUC_train_LR)
    mlflow.sklearn.log_model(my_pipeline, 'model')

    conf_matr = conf_matrix(test_y, class_names, model_result(my_pipeline, train_X, train_y, test_X))

    plot_conf_matrix('Logistic Regression', class_names, conf_matr)

# Cross-validaion

    ROC_AUC_cv_LR = Cr_Val(my_pipeline, train_X, train_y, cv)
    mlflow.log_metric("ROC_AUC_CV_LR", ROC_AUC_cv_LR)


# Test
    model_result(my_pipeline,
             train_X,
             train_y,
             test_X,  
             )

    ROC_AUC_test_LR = round(roc_auc_score(test_y, my_pipeline.predict_proba(test_X)[:, 1]), 2)
    mlflow.log_metric("ROC_AUC_test_LR", ROC_AUC_test_LR)

# Search

    search_space = {
        'clf__penalty': hp.choice('clf__penalty', ['l1', 'l2']),
        'clf__C': hp.uniform(label='clf__C', low = 0.0001, high = 10.0)
    }

    def objective(x):
        params = {
                'clf__penalty': x['clf__penalty'],
                'clf__C': x['clf__C'],
        }

        estimator = my_pipeline.set_params(**params).fit(train_X, train_y)

        score = cross_val_score(estimator = estimator,
                                X = train_X,
                                y = train_y,
                                scoring = 'roc_auc',
                                cv = 10,
                                error_score='raise').mean()
        print(f"AUC {score}, params {params}")

    trials = Trials()

    best = fmin(
            fn = objective,
            space = search_space,
            algo = tpe.suggest,
            max_evals = 100,
            trials = trials,
            rstate = np.random.default_rng(1),
            show_progressbar = True
    )
    
    # penalty_space = ['l1', 'l2']
    best_params = {"penalty": search_space['clf__penalty'][best['clf__penalty']], 'C': best['C']}
    mlflow.log_params(best_params)

# # Best model

    my_pipeline_tuned = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('clf', LogisticRegression(C = best_params['C'], 
                                penalty = best_params['penalty'],
                                solver =  config['solver'],
                                random_state =  config['random_state']))
    ])

    model_result(my_pipeline_tuned,
                train_X,
                train_y,
                test_X,  
                )

    ROC_AUC_test_LR_tuned = round(roc_auc_score(test_y, my_pipeline_tuned.predict_proba(test_X)[:, 1]), 2)
    mlflow.log_metric("ROC_AUC_test_LR_tuned", ROC_AUC_test_LR_tuned)

# Feature importance plots

    feature_importance(my_pipeline_tuned, 20)
    r_a_score(my_pipeline_tuned, train_X, train_y, test_X, test_y)
