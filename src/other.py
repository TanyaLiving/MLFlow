####################################################################################################################################################################################################

## SVM

# Train

my_pipeline_SVM = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', SVC())
])

model_result(my_pipeline_SVM,
             train_X,
             train_y,
             train_X,  
             train_y,
             model_name = 'SVM'
             )

ROC_AUC_train_SVM = roc_auc_score(train_y, my_pipeline_SVM.decision_function(train_X))

# Parameters tuning

tuned_parameters = [{'clf__kernel': ['rbf'], 'clf__gamma': [1e-3, 1e-4],
                     'clf__C': [1, 10, 100, 1000]},]

my_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', SVC())
])

grid_search = GridSearchCV(my_pipeline, param_grid=tuned_parameters,
                           cv=cv, scoring='roc_auc', verbose=2)

grid_search.fit(train_X, train_y)

grid_search.best_params_

# Test

my_pipeline_SVM = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', SVC(C = 1000, kernel = 'rbf', gamma =  0.0001))
])

model_result(my_pipeline_SVM,
             train_X,
             train_y,
             test_X,  
             test_y,
             model_name = 'SVM'
             )

ROC_AUC_test_SVM = roc_auc_score(test_y, my_pipeline_SVM.decision_function(test_X))

r_a_score(my_pipeline_SVM, train_X, train_y, test_X, test_y)

####################################################################################################################################################################################################

## SGDClassifier

sgd_clf = Pipeline([('vectorizer', TfidfVectorizer()),
                    ('clf', SGDClassifier(random_state = 42, shuffle = True, loss = 'modified_huber')),
                    ])

# Train 

model_result(sgd_clf,
             train_X,
             train_y,
             train_X,  
             train_y,
             model_name = 'SGDClassifier'
             )

ROC_AUC_train_SGD = round(roc_auc_score(train_y, sgd_clf.predict_proba(train_X)[:, 1]), 2)

# Cross-validaion

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
ROC_AUC_cv_LR = CV(sgd_clf, train_X, train_y, cv)
scores = cross_val_score(sgd_clf, train_X, train_y, cv=cv, scoring = 'roc_auc')

# Test

model_result(sgd_clf,
             train_X,
             train_y,
             test_X,  
             test_y,
             model_name = 'SGDClassifier'
             )

ROC_AUC_test_SGD = round(roc_auc_score(test_y, sgd_clf.predict_proba(test_X)[:, 1]), 2)

# Parameters tuning

sgd_clf_tuned =  Pipeline([('vectorizer', TfidfVectorizer()),
                           ('clf', SGDClassifier(random_state = 42, shuffle = True, loss = 'modified_huber', penalty = 'elasticnet')),
                          ])

params = {
    "clf__alpha" : [0.0001, 0.001, 0.01, 0.1],
    "clf__l1_ratio" : [0, 0.1, 0.5, 1],
}

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

grid = GridSearchCV(sgd_clf_tuned, param_grid=params, cv=cv, scoring='roc_auc', verbose=2)

grid.fit(train_X, train_y)


# Best model

sgd_clf_best_params = Pipeline([('vectorizer', TfidfVectorizer()),
                    ('clf', SGDClassifier(random_state = 42,  shuffle = True, loss = 'modified_huber', n_jobs = -1, alpha=0.0001, l1_ratio=0.1, penalty='elasticnet')),
                    ])

model_result(sgd_clf_best_params,
             train_X,
             train_y,
             test_X,  
             test_y,
             model_name = 'SGDClassifier'
             )

ROC_AUC_train_SGD_best_params = round(roc_auc_score(test_y, sgd_clf_best_params.predict_proba(test_X)[:, 1]), 2)

# Feature importance

feature_importance(sgd_clf_best_params, 20)

r_a_score(sgd_clf_best_params, 
          train_X,
             train_y,
             test_X,  
             test_y,)

#####################################################################################################################################################################################################

# Refinement

## N-grams

sgd_clf_n_grams = Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1,3) )),
                    ('clf', SGDClassifier(random_state = 42, shuffle = True, loss = 'modified_huber')),
                    ])

model_result(sgd_clf_n_grams,
             train_X,
             train_y,
             test_X,  
             test_y,
             model_name = 'SGDClassifier'
             )

ROC_AUC_test_SGD_ngrams = round(roc_auc_score(test_y, sgd_clf_n_grams.predict_proba(test_X)[:, 1]), 2)

feature_importance(sgd_clf_n_grams, 100)