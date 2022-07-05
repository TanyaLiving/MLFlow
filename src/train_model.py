# from psutil import STATUS_LOCKED
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
from functools import partial

# Models

## Preprocessed data

test_preprocessed = pd.read_csv('/home/asdf/prj/MLFlow/out/test_pipe.csv', sep=';')

train_preprocessed = pd.read_csv('/home/asdf/prj/MLFlow/out/train_pipe.csv', sep=';')

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
    ('clf', LogisticRegression(random_state=1, solver='liblinear'))
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

# # Cross-validaion

cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

ROC_AUC_cv_LR = np.mean(cross_val_score(my_pipeline, X = train_X, y = train_y, cv = cv, scoring = 'roc_auc'))

def Cr_Val(model, X, y, cv):
    CV_score = np.mean(cross_val_score(model, X = X, y = y, cv = cv, scoring = 'roc_auc'))
    return round(CV_score, 2)

ROC_AUC_cv_LR = Cr_Val(my_pipeline, train_X, train_y, cv)
print(f'ROC_AUC_cv_LR_mean_score_train = ', ROC_AUC_cv_LR)


# # Test
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

# presearch

def objective(space):
    params = {
        'penalty': space['clf__penalty'],
        'C': space['clf__C'],
    }
    clf = my_pipeline.set_params(clf__penalty = params['penalty'], clf__C = params['C'])
    score = cross_val_score(estimator = clf,
                            X = train_X,
                            y = train_y,
                            scoring = 'roc_auc',
                            cv = cv,
                            )

    print(f"AUC {score}, params {params}")
    return {'loss': -score.mean(), 
            'params': search_space,
            'status': STATUS_OK}

search_space = {'clf__penalty': hp.choice(label='penalty', options=['l1', 'l2']),
                'clf__C': hp.uniform(label='C', low = 0.0001, high = 100)}


trials = Trials()

best = fmin(
        # optimization function
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals = 100,
        trials = trials,
        rstate = np.random.default_rng(1),
        show_progressbar = True
)

print(f'best - {best}')

# #Search

# search_space = {'clf__penalty': hp.choice(label='penalty', options=['l1', 'l2']),
#                 'clf__C': hp.loguniform(label='C', low=2*np.log(10), high=4*np.log(10))}

# def objective(search_space):
#     '''
#     Cross validation

#     params: dict of parameters
#     pipeline: model
#     X_train: features
#     y_train: labels

#     return: mean score'''

#     # set model's paraneters
#     my_pipeline.set_params(search_space)

#     # cross-validation's parameters
#     cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

#     # cross_validation
#     score = cross_val_score(estimator=pipeline,
#                             X = train_X,
#                             y = train_y,
#                             scoring = 'roc_auc',
#                             cv = cv,
#                             )
#     print("AUC {:.3f} params {}".format(score, search_space))

#     return {'loss': -score.mean(), 
#             'params': search_space,
#             'status': STATUS_OK}

# # for experiment tracking
# trials = Trials()

# best = fmin(
#         # optimization function
#         fn = partial(objective, pipeline = my_pipeline, X_train = train_X, y_train = train_y),
#         space = search_space,
#         algo = tpe.suggest,
#         max_evals = 100,
#         trials = trials,
#         rstate = np.random.default_rng(1),
#         show_progressbar = True
# )

# print(f'best - {best}')
# print(f'trials - {trials.trials}')
# print(f'results - {trials.results}')
# print(f'statuses - {trials.statuses()}')
# print(f'losses - {trials.losses()}')


# # Best model

best = {'C': 2.64158322665802, 'penalty': 1}
penalty_list = ['l1', 'l2', 'elasticnet']

my_pipeline_tuned = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', LogisticRegression(C = best['C'], 
                               penalty = penalty_list[best['penalty']],
                               solver = 'liblinear',
                               random_state = 42))
])

model_result(my_pipeline_tuned,
             train_X,
             train_y,
             test_X,  
             test_y,
             model_name = 'Logistic Regression',
             class_names = class_names
             )

ROC_AUC_test_LR_tuned = round(roc_auc_score(test_y, my_pipeline_tuned.predict_proba(test_X)[:, 1]), 2)
print(f'ROC_AUC_test_LR_tuned = {ROC_AUC_test_LR_tuned}')

# Feature importance plots

feature_importance(my_pipeline_tuned, 20)
print('plot')
r_a_score(my_pipeline_tuned, train_X, train_y, test_X, test_y)
print('r_a_score')
