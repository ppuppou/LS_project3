import pandas as pd

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train.info()
train.head()

train.columns
train = train.drop(columns='id')
train.columns

len(train['작업유형'].unique())