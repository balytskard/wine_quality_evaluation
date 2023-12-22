#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Classifier Building

# ## Exploring the datasets
# 

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df_white = pd.read_csv('data/winequality-white.csv')
df_white.head()


# In[4]:


df_red = pd.read_csv('data/winequality-red.csv')
df_red.head()


# In[5]:


df_white.columns, df_red.columns


# In[6]:


import re


# Splitting all data into appropriate columns
def normalize_dataframe_view(df):
    new_df = df_white.iloc[:, 0].str.split(';', expand=True)
    column_names = df_white.columns.astype(str)[0].split(';')

    quotes = r'"'
    space = r'\s'
    for col_num in range(len(column_names)):
        without_quotation = re.sub(quotes, '', column_names[col_num])
        new_col_name = re.sub(space, '_', without_quotation)
        column_names[col_num] = new_col_name

    new_df.columns = column_names

    return new_df


# In[7]:


data_white, data_red = normalize_dataframe_view(df_white), normalize_dataframe_view(df_red)
data_white.head()


# In[8]:


data_red.head()


# ## Data Processing

# In[9]:


data_white.dtypes, data_red.dtypes


# In[10]:


# Change the data type in dataframes
def change_data_type(df):
    for col in df.columns:
        df[col] = df[col].astype(float)

    return df


data_white, data_red = change_data_type(data_white), change_data_type(data_red)
data_white.dtypes, data_red.dtypes


# In[11]:


data_white.isna().sum(), data_red.isna().sum()


# In[12]:


# Creation one dataframe for both types of the wine
data_white.insert(0, 'color', 'white')
data_red.insert(0, 'color', 'red')
data_wine = pd.concat([data_white, data_red], ignore_index=True)
data_wine


# In[13]:


numerical_features = data_wine.columns[1:-1]
for feature_name in numerical_features:
    max_value = data_wine[feature_name].max()
    min_value = data_wine[feature_name].min()
    data_wine[feature_name] = (data_wine[feature_name] - min_value) / (max_value - min_value)

data_wine.describe()


# In[14]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler


label_encoder = LabelEncoder()
data_wine['color'] = label_encoder.fit_transform(data_wine['color'])

min_max_scaler = MinMaxScaler(feature_range=(1, 10))
data_wine['quality'] = min_max_scaler.fit_transform(data_wine[['quality']])

data_wine.describe()


# Since the total acidity is computed as the sum of the fixed acidity and the volatile acidity, these columns are replaced by a single one.

# In[15]:


data_wine = data_wine.drop('citric_acid', axis=1)
data_wine.insert(1, 'acidity', data_wine['fixed_acidity'] + data_wine['volatile_acidity'])
data_wine = data_wine.drop(['fixed_acidity', 'volatile_acidity'], axis=1)
data_wine.head()


# In[16]:


data_wine.duplicated().sum(), data_wine.shape


# In[17]:


data_wine.drop_duplicates(inplace=True)
data_wine.duplicated().sum()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7))
dataplot = sns.heatmap(data_wine.corr(), annot=True, fmt='.2f')


# In most cases the wine composition on the labels consists only the percentage of alcohol and the sugar, so other features are removed.

# In[19]:


del data_wine['color']
del data_wine['acidity']
del data_wine['chlorides']
del data_wine['free_sulfur_dioxide']
del data_wine['total_sulfur_dioxide']
del data_wine['sulphates']
del data_wine['density']
del data_wine['pH']


# In[20]:


X = data_wine.drop(['quality'], axis=1)
y = data_wine.quality


# In[21]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)


# In[22]:


from imblearn.over_sampling import SMOTE


oversample = SMOTE(random_state=42)
X, y = oversample.fit_resample(X, y)
X.shape, y.shape


# In[23]:


new_min_max_scaler = MinMaxScaler(feature_range=(1, 10))
y = new_min_max_scaler.fit_transform(y.reshape(-1, 1))
y = y.reshape(y.shape[0])
y = y.round(0)
y = y.astype(object)
y


# ## Building Classifier

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[26]:


standard_scaler = StandardScaler()

X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)


# In[27]:


X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


# In[28]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=600, max_features="sqrt")
rf = rf.fit(X_train, y_train)


# In[29]:


y_pred = rf.predict(X_test)


# In[30]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_pred, y_test)
print("Accuracy:", round(accuracy*100, 2), "%")


# # Labels text recognition

# In[31]:


import cv2
import pytesseract
import numpy as np


# In[32]:


def get_composition(image):
    """
    Get text from the wine label photo.
    """

    label_text = pytesseract.image_to_string(image, lang='ukr')

    return label_text


# In[195]:


def find_features(text):
    """
    Find the percentage of alcohol and sugar in the wine.
    """

    patterns = {
        "alcohol" : r"(?i)" + r"(Міцність|спирту:|\nспирту|спирту)" + r"(.{6})",  
        "sugar" : r"(?i)" + r"(цукру|Цукор|\nцукру:|цукру:)" + r"(.{6})"
    }

    matches = {}
    for idx, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            matches[idx] = match.group(2)

    return matches["alcohol"], matches["sugar"]
    


# In[34]:


def get_alcohol_number(alcohol):
    """
    Convert the found alcohol percentage into the number.
    """

    pattern = r"[0-9]+(?:\.[0-9]+)?"
    match = re.search(pattern, alcohol[0])
    if match:
        return float(match.group(0))
    else:
        return 0


# In[35]:


def get_sugar_number(sugar):
    """
    Convert the found sugar percentage into the number
    """
    
    pattern = r"[0-9]+(?:\.[0-9]+)?"
    match = re.search(pattern, sugar[0])
    if match:
        return float(match.group(0))
    else:
        return 0


# In[36]:


def evaluate_quality(image):
    composition = get_composition(image)
    alcohol, sugar = find_features(composition)
    alcohol_number = get_alcohol_number(alcohol)
    sugar_number = get_sugar_number(sugar)

    features = np.array([[alcohol_number, sugar_number]])
    score = rf.predict(features)

    return score[0]


# In[199]:


im = cv2.imread('photos/1.jpg')
print("Wine score: " + str(evaluate_quality(im)) + "/10.0")


# In[ ]:




