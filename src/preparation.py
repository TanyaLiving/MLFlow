import base64
import colorsys
import io
import os
import pickle
import pprint
import re
import string
import sys
import warnings
from collections import Counter
from operator import itemgetter
from random import Random
from xml.sax import saxutils

import contractions
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import unidecode
from bs4 import BeautifulSoup
import nltk
nltk.download('omw-1.4')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from pattern.text.en import singularize
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
from sklearn import datasets, metrics, model_selection, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    plot_confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from spacy import displacy
from spellchecker import SpellChecker
from word2number import w2n
from functions import *

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

warnings.filterwarnings('ignore')

## Read data

data = pd.read_csv('/home/asdf/prj/MLFlow/data/LargeMovieReviewDataset.csv')

## Splitting

X = data.review
y = data.sentiment

sss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_index, test_index = list(sss_train_test.split(X, y))[0]
train = data.loc[train_index]
test = data.loc[test_index]


# Adding information into a dataset

train['review_length'] = train.review.apply(len)
train['number_of_sentences'] = train.review.apply(lambda x: len(sent_tokenize(x)))


# Data prepocessing


train_pipe = pipe(data.loc[train_index])

train_pipe.to_csv('/home/asdf/prj/MLFlow/out/train_pipe.csv', sep=";")

# test pipe

test_pipe = pipe(test)

# Save test processing

test_pipe.to_csv('/home/asdf/prj/MLFlow/out/test_pipe.csv', sep=";")

