import os
import warnings
import yaml
import nltk
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from functions import pipe

nltk.download("omw-1.4")

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")

warnings.filterwarnings("ignore")

config_path = os.path.join("./config/config.yaml")
config = yaml.safe_load(open(config_path))["preparation"]


# for filename in os.listdir("/MLFlow/data/"):
#    with open(os.path.join("/MLFlow/data/", filename), 'r') as f:
#        text = f.read()
#        print(filename)
#        print(text)

# print(os.listdir(f'{os.path.join(os.getcwd(), "data")}'))
# print(config['data_path'])
## Read data
data = pd.read_csv(config["data_path"], sep=",", header=0, nrows=100)
# data = pd.read_csv('/home/asdf/prj/MLFlow/data/LargeMovieReviewDataset.csv', sep=',', header=0, nrows = 1000)
# print(data.sentiment.value_counts())


## Splitting
X = data.review
y = data.sentiment

sss_train_test = StratifiedShuffleSplit(
    n_splits=config["n_splits"],
    test_size=config["test_size"],
    random_state=config["random_state"],
)
# for train_index, test_index in sss_train_test.split(X, y):
#     print("TRAIN:", train_index, "TEST:", test_index)
train_index, test_index = list(sss_train_test.split(X, y))[0]
train = data.loc[train_index]
test = data.loc[test_index]

# Data prepocessing
train_pipe = pipe(data.loc[train_index])
train_pipe.to_csv("/MLFlow/data/train_pipe.csv", sep=";")

# test pipe
test_pipe = pipe(test)

# Save test processing
test_pipe.to_csv("/MLFlow/data/test_pipe.csv", sep=";")