from psutil import STATUS_LOCKED
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
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from functools import patrial

# Models

## Preprocessed data

test_preprocessed = pd.read_pickle('/home/tanya/Education/ML_flow/out/test_pipe.pkl')

train_preprocessed = pd.read_pickle('/home/tanya/Education/ML_flow/out/train_pipe.pkl')

# Splitting

train_X = train_preprocessed.review
train_y = train_preprocessed.sentiment

test_X = test_preprocessed.review
test_y = test_preprocessed.sentiment

class_names = ['negative', 'positive']

#####################################################################################################################################################################

## Logistic regression


# Baseline

my_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', LogisticRegression(random_state=1, n_jobs=-1, solver='liblinear'))
])

# Train
model_result(my_pipeline,
             train_X,
             train_y,
             train_X,  
             train_y,
             model_name = 'Logistic Regression',
             class_names = class_names
             )

ROC_AUC_train_LR = round(roc_auc_score(train_y, my_pipeline.predict_proba(train_X)[:, 1]), 2)
print(f'ROC_AUC train score = {ROC_AUC_train_LR}')

# Cross-validaion

cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

ROC_AUC_cv_LR = np.mean(cross_val_score(my_pipeline, X = train_X, y = train_y, cv = cv, scoring = 'roc_auc'))

def Cr_Val(model, X, y, cv):
    CV_score = np.mean(cross_val_score(model, X = X, y = y, cv = cv, scoring = 'roc_auc'))
    return round(CV_score, 2)

ROC_AUC_cv_LR = Cr_Val(my_pipeline, train_X, train_y, cv)
print(f'ROC_AUC_cv_LR_mean_score_train = ', ROC_AUC_cv_LR)


# Test
model_result(my_pipeline,
             train_X,
             train_y,
             test_X,  
             test_y,
             model_name = 'Logistic Regression',
             class_names = class_names
             )

ROC_AUC_test_LR = round(roc_auc_score(test_y, my_pipeline.predict_proba(test_X)[:, 1]), 2)
print(f'ROC_AUC_cv_LR_mean_score_test =  {ROC_AUC_test_LR}')

search_space = {'lr__penalty': hp.choice(label='penalty', options=['l1', 'l2']),
                'lt__C': hp.logiuniform(label='C', low=4*np.log(10), high=2*np.log(10))}

def objective(params, pipeline, X_train, y_train):
    '''
    Cross validation

    params: dict of parameters
    pipeline: model
    X_train: features
    y_train: labels

    return: mean score'''

    # set model's paraneters
    pipeline.set_params(**params)

    # cross-validation's parameters
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42, shuffle=True)

    # cross_validation
    score = cross_val_score(estimator=pipeline,
                            X = X_train,
                            y = y_train,
                            scoring = 'roc_auc',
                            cv = cv,
                            n_jobs=-1)

    return {'loss': -score.mean(), 
            'params': params,
            'status': STATUS_OK}

# for experiment tracking
trials = Trials()

best = fmin(
        # optimization function
        fn = patrial(objective, pipeline = my_pipeline, X_train = train_X, y_train = train_y),
        space = search_space,
        algo = tpe.suggest,
        max_evals=100,
        trials = trials,
        rstate = np.RandomState(1),
        show_progressbar = True
)

def search_results():
    
# # Parameters tuning

# my_pipeline_tuning = Pipeline([
#     ('vectorizer', TfidfVectorizer()),
#     ('clf', LogisticRegression(penalty = 'elasticnet',
#                                          solver = 'saga',
#                                           random_state = 42,
#                                           verbose=1))
# ])


# # tuned_parameters = [{'clf__C' :  np.linspace(1e-3, 1e3),
# #                      'clf__l1_ratio': [0, 0.1, 0.5, 1]
# #                     }]
# # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

# # grid_search = GridSearchCV(my_pipeline_tuning, param_grid=tuned_parameters,
# #                            cv=cv, scoring='roc_auc', verbose=2)
# # grid_search.fit(train_X, train_y)
# # print(f'best_params -  {grid_search.best_params_}')


# # # Best model

# my_pipeline_tuned = Pipeline([
#     ('vectorizer', TfidfVectorizer()),
#     ('clf', LogisticRegression(C = 1.0002302850208247, 
#                                l1_ratio = 0,
#                                penalty = 'elasticnet',
#                                solver = 'elasticnet',
#                                random_state = 42))
# ])

# model_result(my_pipeline_tuned,
#              train_X,
#              train_y,
#              test_X,  
#              test_y,
#              model_name = 'Logistic Regression',
#              class_names = class_names
#              )

# ROC_AUC_test_LR_tuned = round(roc_auc_score(test_y, my_pipeline_tuned.predict_proba(test_X)[:, 1]), 2)

# # Feature importance plots

# feature_importance(my_pipeline_tuned, 20)

# r_a_score(my_pipeline_tuned, train_X, train_y, test_X, test_y)

