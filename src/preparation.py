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
import yaml

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
import nltk
nltk.download('omw-1.4')
from nltk.tokenize import sent_tokenize, word_tokenize
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from sklearn.model_selection import StratifiedShuffleSplit
from functions import *

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

warnings.filterwarnings('ignore')

config_path = os.path.join('/home/asdf/prj/MLFlow/config/config.yaml')
config = yaml.safe_load(open(config_path))['preparation']

## Read data

data = pd.read_csv(config['data_path'])

## Splitting

X = data.review
y = data.sentiment

sss_train_test = StratifiedShuffleSplit(n_splits=config['n_splits'], test_size=config['test_size'], random_state=config['random_state'])
train_index, test_index = list(sss_train_test.split(X, y))[0]
train = data.loc[train_index]
test = data.loc[test_index]

# Data prepocessing

train_pipe = pipe(data.loc[train_index])

train_pipe.to_csv('/home/asdf/prj/MLFlow/out/train_pipe.csv', sep=";")

# test pipe

test_pipe = pipe(test)

# Save test processing

test_pipe.to_csv('/home/asdf/prj/MLFlow/out/test_pipe.csv', sep=";")

