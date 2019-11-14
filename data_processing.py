import pandas as pd
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from classifiers import *
import time

# Switch between the two cleaned datasets
df = pd.read_pickle('clean_df')
#df = pd.read_pickle("clean_df_v2")



# Lets you reference the top 100 words across all essays, I chose the ones that I thought might be most helpful.
#essay_word_count = dict(Counter(" ".join(all_essays).split()).most_common(100))
select_words = ['time', 'people', 'books', 'friends', 'working', 'love', 'my', 'fun']
for word in select_words:
    df['essay_{}'.format(word)] = df.all_essays.apply(lambda row: Counter(str(row).split())[word])
# Finds len of all essays
#df["essay_len"] = all_essays.apply(lambda x: len(x)).reset_index()

# Select your features
selected_features = [
    'diet_code',
    'drinks_code',
    'smokes_code',
    'education_code',
    'drugs_code',
    'income_reported',
#    'essay_time',
#    'essay_people',
#    'essay_books',
#    'essay_friends',
#    'essay_working',
#    'essay_love',
#    'essay_my',
#    'essay_fun'
    ]


# iterates through selected labels and spits out the best scores for different classifiers
selected_labels = ['body_type']
for selected_label in selected_labels:
    feature_data = df[selected_features + [selected_label]].dropna(axis=0)

    labels = np.array(feature_data[selected_label])
    features = feature_data[selected_features]

    x = features.values
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(x)

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=23)

    print("|| " + selected_label.upper().replace('_', ' ') + " LABEL ||\n")
    t0 = time.time()
    print(decision_tree(features_train, labels_train, features_test, labels_test))
    print("Time to run = {}\n".format(time.time() - t0))
    t0 = time.time()
    print(random_forest(features_train, labels_train, features_test, labels_test))
    print("Time to run = {}\n".format(time.time() - t0))
    t0 = time.time()
    print(k_nearest_neighbor(features_train, labels_train, features_test, labels_test))
    print("Time to run = {}\n".format(time.time() - t0))

    print(k_nearest_neighbor_graph(features_train, labels_train, features_test, labels_test))
    print(decision_tree_graph(features_train, labels_train, features_test, labels_test))


# Regression Models

# Linear Regression
df['income_under_100k'] = df[df['income_reported'] < 100000]['income_reported']
df.dropna(how='any', subset=['education_code', 'income_under_100k'], inplace=True)
x = np.array(df['education_code']).reshape(-1, 1)
y = np.array(df.income_under_100k)
t0 = time.time()
print("Linear Regression Score = {}".format(linear_regression(x, y)))
print("Time = {}".format(time.time() - t0))

# Multiple Linear Regression
selected_features = ['education_code', 'essay_length']
selected_label = 'income_under_100k'
feature_data = df[selected_features + [selected_label]].dropna(axis=0)

labels = np.array(feature_data[selected_label])
features = feature_data[selected_features]

x = features.values
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=3)
t0 = time.time()
print("Multi Linear Regression Score = {}".format(multi_linear_regression(x_train, y_train, x_test, y_test)))
print("Time = {}".format(time.time() - t0))