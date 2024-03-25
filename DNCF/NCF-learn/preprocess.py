import random
from functools import lru_cache
import os
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder

# Hack for running on kernels and locally
RUNNING_ON_KERNELS = 'KAGGLE_WORKING_DIR' in os.environ
input_dir = '../NCF-Data/otherlens'
out_dir = '.' if RUNNING_ON_KERNELS else '../NCF-Data/movielens_preprocessed'

rating_path = os.path.join(input_dir, 'rating.csv')
df = pd.read_csv(rating_path, usecols=['userId', 'movieId', 'rating'])
# Shuffle (reproducibly)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

# Partitioning train/val according to behaviour of keras.Model.fit() when called with
# validation_split kwarg (which is to take validation data from the end as a contiguous
# chunk)
val_split = .05
n_ratings = len(df)
n_train = math.floor(n_ratings * (1-val_split))
itrain = df.index[:n_train]
ival = df.index[n_train:]

# Compactify movie ids.
movie_id_encoder = LabelEncoder()
# XXX: Just fitting globally for simplicity. See movie_helpers.py for more 'principled'
# approach. I don't think there's any realistically useful data leakage here though.
#orig_movieIds = df['movieId']
df['movieId'] = movie_id_encoder.fit_transform(df['movieId'])

# Add centred target variable
df['y'] = df['rating'] - df.loc[itrain, 'rating'].mean()

SCALE = 0
if SCALE:
    # Add version of target variable scale to [0, 1]
    yscaler = sklearn.preprocessing.MinMaxScaler()
    yscaler.fit(df.loc[itrain, 'rating'].values.reshape(-1, 1))
    df['y_unit_scaled'] = yscaler.transform(df['rating'].values.reshape(-1, 1))

path = os.path.join(out_dir, 'rating.csv')
df.to_csv(path, index=False)

# Save a 10% sample of ratings for exercises (with re-compactified movieIds, and mapping back to canonical movie ids)
from sklearn.model_selection import GroupShuffleSplit

movie_counts = df.groupby('movieId').size()
thresh = 1000
pop_movies = movie_counts[movie_counts >= thresh].index

pop_df = df[df.movieId.isin(pop_movies)]

# Take approx 10% of the whole dataset
frac = 2 * 10 ** 6 / len(pop_df)
print(frac)
splitter = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=1)
splits = splitter.split(pop_df, groups=pop_df.userId)
_, mini = next(splits)

mini_df = pop_df.iloc[mini].copy()

print(
    '{:,}'.format(len(mini_df)),
    len(df.userId.unique()) // 1000,
    len(mini_df.userId.unique()) // 1000,
    sep='\n',
)


# Compactify ids

def compactify_ids(df, col, backup=True):
    encoder = LabelEncoder()
    if backup:
        df[col + '_orig'] = df[col]
    df[col] = encoder.fit_transform(df[col])


for col in ['movieId', 'userId']:
    compactify_ids(mini_df, col, backup=col == 'movieId')

# Shuffle
mini_df = mini_df.sample(frac=1, random_state=1)

# Recalculate y (just to be totally on the level. Very little opportunity for contamination here.)
val_split = .05
n_mini_train = math.floor(len(mini_df) * (1 - val_split))
mini_train_rating_mean = mini_df.iloc[:n_mini_train]['rating'].mean()
mini_df['y'] = mini_df['rating'] - mini_train_rating_mean

path = os.path.join(out_dir, 'mini_rating.csv')
mini_df.to_csv(path, index=False)

print(df.userId.max(),
    mini_df.userId.max(),
    '\n',
    df.movieId.max(),
    mini_df.movieId.max())