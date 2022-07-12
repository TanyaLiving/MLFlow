import re
import contractions
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import unidecode
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from spellchecker import SpellChecker
import seaborn as sns
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from nltk.tokenize import sent_tokenize, word_tokenize
import mlflow

## Auxiliary methods

def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café, Renée Zellweger"""
    text = unidecode.unidecode(text)
    return text

def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text

def model_result(model, X_tran, y_train, X_test):
  mod = model.fit(X_tran, y_train)
  y_pred = mod.predict(X_test)
  # cvs = cross_val_score(mod, X_tran, y_train, cv = cv, scoring = 'roc_auc')

  return y_pred

def conf_matrix(y_test, class_names, model_result):
  conf_matrix = pd.DataFrame(confusion_matrix(y_test, model_result), index = class_names, columns = class_names)
  conf = confusion_matrix(y_test, model_result)

  return conf

def plot_conf_matrix(model_name, class_names, conf_matrix):
  fig = plt.figure(figsize=(12, 4))
  plt.subplots_adjust(wspace=0.5, hspace=2) 
  plt.subplot(1, 2, 1) # nrow 1 ncol 2 index 1 
  plt.title(f'{model_name} confusion matrix', fontdict={'fontsize':13}, pad=12, fontweight= 'semibold');
  sns.heatmap(conf_matrix/np.sum(conf_matrix, axis=1).reshape(-1,1),
              annot=conf_matrix/np.sum(conf_matrix, axis=1).reshape(-1,1),
              annot_kws={"size": 10},
              fmt='.2f',
              yticklabels=class_names,
              xticklabels=class_names,
              cmap='YlGn',
              cbar_kws={"shrink": .82},
              linewidths=0.2, 
              linecolor='gray'
              );
  plt.xlabel('Predicted label', fontsize = 11);
  plt.ylabel('True label', fontsize = 11);
  plt.xticks(fontsize = 9)   
  plt.yticks(fontsize = 9)


  plt.subplot(1, 2, 2)
  sns.heatmap(conf_matrix, annot=True,
                    annot_kws={"size": 10},
                    yticklabels=class_names,
                    xticklabels=class_names,
                    fmt='d', cmap='YlGn', 
                    cbar_kws={"shrink": .82},
                    linewidths=0.2, 
                    linecolor='gray')

  plt.title(f'{model_name} confusion matrix', fontdict={'fontsize':13}, pad=12, fontweight= 'semibold');
  plt.xlabel('Predicted label', fontsize = 11);
  plt.ylabel('True label', fontsize = 11);
  plt.xticks(fontsize = 9)   
  plt.yticks(fontsize = 9)


  # save plot
  # plt.savefig('confusion_matrix_plot.png')
  mlflow.log_figure(fig, artifact_file = 'plots/confusion_matrix_plot.png')

def Cr_Val(model, X, y, cv):
    CV_score = np.mean(cross_val_score(model, X = X, y = y, cv = cv, scoring = 'roc_auc'))
    return round(CV_score, 2)

def feature_importance(clf, n_features):
  feature_names = clf.named_steps["vectorizer"].get_feature_names_out()
  coefs = clf.named_steps['clf'].coef_.flatten()

  # Zip coefficients and names together and make a DataFrame
  zipped = zip(feature_names, coefs)
  df = pd.DataFrame(zipped, columns=["feature", "value"])

  # Sort the features by the absolute value of their coefficient
  df["abs_value"] = df["value"].apply(lambda x: abs(x))
  df["colors"] = df["value"].apply(lambda x: "#316879" if x > 0 else "#f47a60")
  df = df.sort_values(["abs_value", 'colors'], ascending=False)

  max_green = df.loc[df.colors == '#f47a60'][:n_features]
  max_red = df.loc[df.colors == '#316879'][:n_features]
  fi = pd.concat([max_green, max_red])

  fig, ax = plt.subplots(1, 2, figsize=(25, n_features/2))
  plt.subplots_adjust(wspace=0.4, hspace=0.6)
  plt.subplot(1, 2, 1) # nrow 1 ncol 2 index 1 
  sns.set (style = "whitegrid")
  sns.barplot(x="abs_value",
              y="feature",
              data=fi[:n_features],
              color = '#316879');

  plt.title(f"Top {n_features} Features negative review", fontsize=25,  color='#4f4e4e', fontweight = 'demibold');
  plt.xlabel("Absolute coefficients", fontsize=20,  color='#4f4e4e');
  plt.ylabel("Feature names", fontsize=20,  color='#4f4e4e');
  sns.despine ();
  plt.xticks (size = 20,  color='#4f4e4e');
  plt.yticks (size = 20,  color='#4f4e4e');
  neg = mlines.Line2D([], [], color='#f47a60', marker='s', linestyle='None',
                            markersize=10, label='negative');
  pos = mlines.Line2D([], [], color='#316879', marker='s', linestyle='None',
                            markersize=10, label='positive');

  plt.subplot(1, 2, 2)
  sns.barplot(x="abs_value",
              y="feature",
              data=fi[n_features:],
              color = '#f47a60');
  plt.title(f"Top {n_features} features positive review", fontsize=25,  color='#4f4e4e', fontweight = 'demibold');
  plt.xlabel("Absolute coefficients", fontsize=20,  color='#4f4e4e');
  plt.ylabel("Feature names", fontsize=20,  color='#4f4e4e');
  sns.despine ();
  plt.xticks (size = 20,  color='#4f4e4e');
  plt.yticks (size = 20,  color='#4f4e4e');
  neg = mlines.Line2D([], [], color='#f47a60', marker='s', linestyle='None',
                            markersize=10, label='negative');
  pos = mlines.Line2D([], [], color='#316879', marker='s', linestyle='None',
                            markersize=10, label='positive');

  plt.legend(handles=[neg, pos], loc = 2, bbox_to_anchor = (1,0.55), fontsize = 16);

def r_a_score(clf, train_X, train_y, test_X, test_y):
  
  try:
    y_pred_prob_train = clf.predict_proba(train_X)[:,1]
    y_pred_prob_test = clf.predict_proba(test_X)[:,1]

  except:
    y_pred_prob_train = clf.decision_function(train_X)
    y_pred_prob_test = clf.decision_function(test_X)

  fpr1 , tpr1, thresholds1 = roc_curve(train_y, y_pred_prob_train)
  fpr2 , tpr2, thresholds2 = roc_curve(test_y, y_pred_prob_test)

  ROC_AUC_train = round(roc_auc_score(train_y, y_pred_prob_train), 2)
  ROC_AUC_test = round(roc_auc_score(test_y, y_pred_prob_test), 2)

  plt.figure(figsize=(12, 8))
  plt.plot([0,1],[0,1], 'k--')
  plt.plot(fpr1, tpr1, label= f"Train (AUC = {ROC_AUC_train})")
  plt.plot(fpr2, tpr2, label= f"Test (AUC = {ROC_AUC_test})")
  plt.legend(fontsize = 16)
  plt.xlabel("False Positive Rate", fontsize=20,  color='#4f4e4e')
  plt.ylabel("True Positive Rate", fontsize=20,  color='#4f4e4e')
  plt.title('Receiver Operating Characteristic for the best model', fontsize=25,  color='#4f4e4e', fontweight = 'demibold')
  plt.xticks (size = 20,  color='#4f4e4e');
  plt.yticks (size = 20,  color='#4f4e4e');
  plt.show()

  ## Pipeline

def pipe(df):

  col = df.columns[0]
  col_y = df.columns[1]
  tokenizer = RegexpTokenizer(r"[A-Za-z]+")
  list_to_remove = ['no', 'not']
  stop_words = list(set((stopwords.words('english'))).difference(set(list_to_remove)))
  NER = spacy.load("en_core_web_sm")
  spell = SpellChecker(language = 'en' , local_dictionary = None , distance = 2)
  sb_stemmer = SnowballStemmer("english")
  WNLemmatizer = WordNetLemmatizer()

  """Number_of_sentences"""
  df['number_of_sentences'] = df.review.apply(lambda x: len(sent_tokenize(x)))

  """Review's len"""
  df['review_length'] = df.review.apply(len)

  """Remove HTML Tags"""
  df[col] = df[col].apply(lambda x: strip_html_tags(x))
  
  """Remove URLs"""
  df[col] = df[col].apply(lambda x: re.sub(r'((https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}))',
                                    '',
                                    x,
                                    flags=re.IGNORECASE))
  
  """Remove repeated letters"""
  df[col] = df[col].apply(lambda x: re.sub(r'(\w)\1{2,}', r'\1', x))
  
  """Remove whitespaces"""
  df[col] = df[col].apply(lambda x: x.strip ()) 

  """Convert Accented Characters"""
  df[col] = df[col].apply(lambda x: remove_accented_chars(x)) 

  """Expand Contractions"""
  df[col] = df[col].apply(lambda x: expand_contractions(x))

  """Remove numbers"""
  df[col] = df[col].apply(lambda x: re.sub(r'\d+', '', x))

  """Remove punctuation + tokenize"""
  df[col] = df[col].apply(lambda x: tokenizer.tokenize(x))

  """Stopwords"""
  df[col] = df[col].apply(lambda x: [word for word in x if word.lower() not in stop_words])

  # """Singularize"""
  # df[col] = df[col].apply(lambda x: [singularize(i) for i in x])

  """Named entity"""
  df['NE'] = df[col].apply(lambda x: [ent.text for ent in NER(' '.join(x)).ents])

  """Remove NE"""
  df[col] = df.apply(lambda x: [word for word in x[col] if word not in set(tokenizer.tokenize(' '.join(x['NE'])))], axis=1)

  """Misspelling"""
  df[col] = df[col].apply(lambda x: [spell.correction(word) for word in x])

  """Stop - words 2"""
  df[col] = df[col].apply(lambda x: [word for word in x if word.lower() not in stop_words])

  """Convert text to lowercase"""
  df[col] = df[col].apply(lambda x: [word.lower() for word in x])

  """Lemmatizer"""
  df[col] = df[col].apply(lambda x: [WNLemmatizer.lemmatize(word) for word in x])

  """Preprocessing for Vectorizer remove shortest words"""
  df[col] = df[col].apply(lambda x: ' '.join([word for word in x if len(word)>1]))

  """Target prprocesssing"""
  df[col_y] = df[col_y].replace({'positive': 1, 'negative': 0})

  return df

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


def Cr_Val(model, X, y, cv):
    CV_score = np.mean(cross_val_score(model, X = X, y = y, cv = cv, scoring = 'roc_auc'))
    return round(CV_score, 2)
