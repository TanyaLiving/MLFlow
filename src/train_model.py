

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

config_path = os.path.join('/home/asdf/prj/MLFlow/config/config.yaml')
config = yaml.safe_load(open(config_path))['train_model']

## Preprocessed data

test_preprocessed = pd.read_csv(config['data_path_test'], sep=';')

train_preprocessed = pd.read_csv(config['data_path_train'], sep=';')

# Splitting

train_X = train_preprocessed.review
train_y = train_preprocessed.sentiment

test_X = test_preprocessed.review
test_y = test_preprocessed.sentiment

class_names = ['negative', 'positive']


# client = MlflowClient()
# for i in client.list_experiments():
#     print(i)

# #####################################################################################################################################################################
# # mlflow.set_tracking_uri('/home/asdf/prj/MLFlow/src/mlruns')
# # client = MlflowClient()
# # # mlflow.set_experiment("/my-experiment")
# # mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# # # print(type(client.get_experiment_by_name('First').experiment_id))
# client.delete_experiment(client.get_experiment_by_name('first_track').experiment_id)
mlflow.set_experiment(experiment_name = config["experiment_name"])
# # mlflow.set_tags({'tag_name': "TEST_EXP"})
# # mlflow.sklearn.autolog()

## Logistic regression


# Baseline

params = {"random_state": config['random_state'], 'solver': config['solver']}

cv = StratifiedShuffleSplit(n_splits=config['n_splits'], test_size=config['test_size'], random_state=config['random_state'])
# vectorizer = TfidfVectorizer()
# vector_data_train = vectorizer.fit_transform(train_X)
# lr = LogisticRegression()

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



    # # Model registry does not work with file store
    # if tracking_url_type_store != "file":

    #     # Register the model
    #     # There are other ways to use the Model Registry, which depends on the use case,
    #     # please refer to the doc for more information:
    #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    #     mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
    # else:
    #     mlflow.sklearn.log_model(lr, "model")

# # Cross-validaion

# ROC_AUC_cv_LR = np.mean(cross_val_score(my_pipeline, X = train_X, y = train_y, cv = cv, scoring = 'roc_auc'))

# ROC_AUC_cv_LR = Cr_Val(my_pipeline, train_X, train_y, cv)
# print(f'ROC_AUC_cv_LR_mean_score_train = ', ROC_AUC_cv_LR)


# # # Test
# model_result(my_pipeline,
#              train_X,
#              train_y,
#              test_X,  
#              test_y,
#              model_name = 'Logistic Regression',
#              class_names = class_names,
#              title = 'Logistic Regression'
#              )

# ROC_AUC_test_LR = round(roc_auc_score(test_y, my_pipeline.predict_proba(test_X)[:, 1]), 2)
# print(f'ROC_AUC_cv_LR_mean_score_test =  {ROC_AUC_test_LR}')

# # Search

# trials = Trials()

# best = fmin(
#         # optimization function
#         fn = objective,
#         space = search_space,
#         algo = tpe.suggest,
#         max_evals = 100,
#         trials = trials,
#         rstate = np.random.default_rng(1),
#         show_progressbar = True
# )

# print(f'best - {best}')

# # # Best model

# best = {'C': 2.64158322665802, 'penalty': 1}
# penalty_list = ['l1', 'l2', 'elasticnet']

# my_pipeline_tuned = Pipeline([
#     ('vectorizer', TfidfVectorizer()),
#     ('clf', LogisticRegression(C = best['C'], 
#                                penalty = penalty_list[best['penalty']],
#                                solver = 'liblinear',
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
# print(f'ROC_AUC_test_LR_tuned = {ROC_AUC_test_LR_tuned}')

# # Feature importance plots

# feature_importance(my_pipeline_tuned, 20)
# print('plot')
# r_a_score(my_pipeline_tuned, train_X, train_y, test_X, test_y)
# print('r_a_score')
