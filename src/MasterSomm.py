'''
Nirav Shah
CMSC 478
Prof. Marron

Note: I used code from a Kaggle kernel as a basis for my project. Here is the source: https://www.kaggle.com/carkar/classifying-wine-type-by-review/notebook
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import unicodedata
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/anila/PycharmProjects/WineData150.csv")

#Removing duplicates
data[data.duplicated('description', keep=False)].sort_values('description').head(5)
data = data.drop_duplicates('description')
data = data[pd.notnull(data.price)]
#print(data.shape)

#Filtering out stuff
country = data.groupby('country').filter(lambda x: len(x) >100)

X = data.drop(['Unnamed: 0','country','designation','points','price','province','region_1','region_2','variety','winery'], axis = 1)
y = data.variety

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Only counting wine varieties with >> 200 observations
data = data.groupby('variety').filter(lambda x: len(x) >200)

#Replacing German names with English names
data['variety'] = data['variety'].replace(['Weissburgunder'], 'Chardonnay')
data['variety'] = data['variety'].replace(['Spatburgunder'], 'Pinot Noir')
data['variety'] = data['variety'].replace(['Grauburgunder'], 'Pinot Gris')

#Replace Spanish garnacha with French grenache
data['variety'] = data['variety'].replace(['Garnacha'], 'Grenache')

#Replace Italian pinot nero with French pinot noir
data['variety'] = data['variety'].replace(['Pinot Nero'], 'Pinot Noir')

#Replace Portuguese alvarinho with Spanish albarino
data['variety'] = data['variety'].replace(['Alvarinho'], 'Albarino')

#Get rid of blends and roses
data = data[data.variety.str.contains('blend') == False]
data = data[data.variety.str.contains('rose') == False]

'''
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


data['variety'] = data['variety'].apply(remove_accents)
data['description'] = data['description'].apply(remove_accents)'''

#Cleaning up our varieties to make life easier
wine = data.variety.unique().tolist()
wine.sort()

output = set()
for x in data.variety:
    x = x.lower()
    x = x.split()
    for y in x:
        output.add(y)

variety_list =sorted(output)

extras = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', 'cab',"%"]
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(variety_list)
stop.update(extras)

from scipy.sparse import hstack

#Creating our training and test data
vect = CountVectorizer(stop_words = stop)
X_train_dtm = vect.fit_transform(X_train.description)
#price = X_train.price.values[:,None]
#X_train_dtm = hstack((X_train_dtm, price))

X_test_dtm = vect.transform(X_test.description)
#price_test = X_test.price.values[:,None]
#X_test_dtm = hstack((X_test_dtm, price_test))

#Setting a seed so we can get consistent results
seed = 7


#Generating the logistic models using our training data
models = {}
for z in wine:
    model = LogisticRegression(random_state=seed)
    y = y_train == z
    model.fit(X_train_dtm, y)
    models[z] = model

#Using 10-fold cross validation to generate a score for model comparison
testModel = LogisticRegression(random_state=seed)
scoresLog = cross_val_score(testModel, X_train_dtm, y, cv=10)
print("Accuracy: %0.6f (+/- %0.6f)" % (scoresLog.mean(), scoresLog.std() * 2))

testing_probs = pd.DataFrame(columns=wine)

#This is running the logistic models on our test data
for variety in wine:
    testing_probs[variety] = models[variety].predict_proba(X_test_dtm)[:, 1]

predicted_wine = testing_probs.idxmax(axis=1)

comparison = pd.DataFrame({'actual': y_test.values, 'predicted': predicted_wine.values})

#Printing the accuracy
print('Logistic Regression Accuracy Score:',  accuracy_score(comparison.actual, comparison.predicted)*100, "%")
print(comparison.head(20))


#Generating the Random Forest Classifier models using our training data
modelsRFC = {}
for z in wine:
    modelRFC = RandomForestClassifier(random_state=seed, n_estimators=25, n_jobs=1)
    y = y_train == z
    modelRFC.fit(X_train_dtm, y)
    modelsRFC[z] = modelRFC

#Using 10-fold cross validation to generate a score for model comparison
testModel = RandomForestClassifier(random_state=seed, n_estimators=25, n_jobs=1)
scoresRFC = cross_val_score(testModel, X_train_dtm, y, cv=10)
print("Accuracy: %0.6f (+/- %0.6f)" % (scoresRFC.mean(), scoresRFC.std() * 2))
testModel.fit(X_train_dtm, y)
importances = testModel.feature_importances_

testing_probs_RFC = pd.DataFrame(columns = wine)

#This is running the Random Forest Classifer models on our test data
for variety in wine:
    testing_probs_RFC[variety] = modelsRFC[variety].predict_proba(X_test_dtm)[:, 1]

predicted_wine_RFC = testing_probs_RFC.idxmax(axis=1)

comparisonRFC = pd.DataFrame({'actual': y_test.values, 'predicted': predicted_wine_RFC.values})

#Printing the accuracy
print('Random Forest Classifier Accuracy Score:', accuracy_score(comparisonRFC.actual, comparisonRFC.predicted) * 100, "%")
print(comparisonRFC.head(20))

std = np.std([tree.feature_importances_ for tree in testModel.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train_dtm.shape[0]):
    print("%d. feature %d (%f)" % (f, indices[f], importances[indices[f]]))

print("Testing: ", vect.get_feature_names())


#Generating the Ada Boost Classifier models using our training data
modelsADA = {}
for z in wine:
    modelADA = AdaBoostClassifier(random_state=seed, n_estimators=200, learning_rate=0.5)
    y = y_train == z
    modelADA.fit(X_train_dtm, y)
    modelsADA[z] = modelADA

#Using 10-fold cross validation to generate a score for model comparison
testModel = AdaBoostClassifier(random_state=seed, n_estimators=200, learning_rate=0.5)
scoresADA = cross_val_score(testModel, X_train_dtm, y, cv=10)
print("Accuracy: %0.6f (+/- %0.6f)" % (scoresADA.mean(), scoresADA.std() * 2))

testing_probs_ADA = pd.DataFrame(columns = wine)

#This is running the Ada Boost Classifer models on our test data
for variety in wine:
    testing_probs_ADA[variety] = modelsADA[variety].predict_proba(X_test_dtm)[:, 1]

predicted_wine_ADA = testing_probs_ADA.idxmax(axis=1)

comparisonADA = pd.DataFrame({'actual': y_test.values, 'predicted': predicted_wine_ADA.values})

#Printing the accuracy
print('Ada Boost Classifier Accuracy Score:', accuracy_score(comparisonADA.actual, comparisonADA.predicted) * 100, "%")
print(comparisonADA.head(20))


#Generating the Extra Trees Classifier models using our training data
modelsETC = {}
for z in wine:
    modelETC = ExtraTreesClassifier(random_state=seed, n_estimators=25, n_jobs=1)
    y = y_train == z
    modelETC.fit(X_train_dtm, y)
    modelsETC[z] = modelETC

#Using 10-fold cross validation to generate a score for model comparison
testModel = ExtraTreesClassifier(random_state=seed, n_estimators=25, n_jobs=1)
scoresETC = cross_val_score(testModel, X_train_dtm, y, cv=10)
print("Accuracy: %0.6f (+/- %0.6f)" % (scoresETC.mean(), scoresETC.std() * 2))
testModel.fit(X_train_dtm, y)
print(testModel.feature_importances_)

testing_probs_ETC = pd.DataFrame(columns = wine)

#This is running the Extra Trees Classifer models on our test data
for variety in wine:
    testing_probs_ETC[variety] = modelsETC[variety].predict_proba(X_test_dtm)[:, 1]

predicted_wine_ETC = testing_probs_ETC.idxmax(axis=1)

comparisonETC = pd.DataFrame({'actual': y_test.values, 'predicted': predicted_wine_ETC.values})

#Printing the accuracy
print('Extra Trees Classifier Accuracy Score:', accuracy_score(comparisonETC.actual, comparisonETC.predicted) * 100, "%")
print(comparisonETC.head(20))


#Generating the Gradient Boosting Classifier models using our training data
modelsGBC = {}
for z in wine:
    modelGBC = GradientBoostingClassifier(random_state=seed, n_estimators=200, learning_rate=0.1)
    y = y_train == z
    modelGBC.fit(X_train_dtm, y)
    modelsGBC[z] = modelGBC

#Using 10-fold cross validation to generate a score for model comparison
testModel = GradientBoostingClassifier(random_state=seed, n_estimators=200, learning_rate=0.1)
scoresGBC = cross_val_score(testModel, X_train_dtm, y, cv=10)
print("CV Accuracy Score: %0.6f (+/- %0.6f)" % (scoresGBC.mean(), scoresGBC.std() * 2))


testing_probs_GBC = pd.DataFrame(columns = wine)

#This is running the Gradient Boosting Classifer models on our test data
for variety in wine:
    testing_probs_GBC[variety] = modelsGBC[variety].predict_proba(X_test_dtm)[:, 1]

predicted_wine_GBC = testing_probs_GBC.idxmax(axis=1)

comparisonGBC = pd.DataFrame({'actual': y_test.values, 'predicted': predicted_wine_GBC.values})

#Printing the accuracy
print('Gradient Boosting Classifier Accuracy Score:', accuracy_score(comparisonGBC.actual, comparisonGBC.predicted) * 100, "%")
print(comparisonGBC.head(20))


