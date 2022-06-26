# -*- coding: utf-8 -*-
"""sentimentanalysis.ipynb


the whole is done using GOOGLE COLAB
My Original  file is located at
    https://colab.research.google.com/gist/Mr-KPK/455b38353159367251224ea78f50fbc3/sentimentanalysis.ipynb

**create a dataframe**
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
df = pd.read_csv('/content/Restaurant_Reviews.tsv', delimiter='\t')
df.head()

"""**process the data**"""

len(df)

df.isnull().sum()

df.shape

df.info()

df['Review'].value_counts

df['Liked'].value_counts

"""**comparing positive reviews and negative reviews**"""

sns.countplot(df['Liked'])

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
df['Review'][0]

corpus = []
for i in range(0,1000):
  review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
  review = review.lower()
  review = review.split()
  stemmer = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [stemmer.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

df['cleaned_text']=corpus
df.head()

"""**testing and training  the data**"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df['cleaned_text'],df['Liked'],test_size=0.2,random_state=0)

"""**count vectorizer**"""

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1420)
X_train_1 = cv.fit_transform(X_train).toarray()
X_test_1 = cv.transform(X_test).toarray()

"""**navie bayes**"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix , accuracy_score

Bayes_classifier = GaussianNB()
Bayes_classifier.fit(X_train_1,y_train)
y_pred = Bayes_classifier.predict(X_test_1)
cn = confusion_matrix(y_test ,y_pred)
print(confusion_matrix(y_test,y_pred))
sns.heatmap(cn , annot=True)
accuracy_score(y_test,y_pred)

"""**svc model**"""

from sklearn.svm import SVC
svc_classifier = SVC(kernel='rbf')
svc_classifier.fit(X_train_1,y_train)
svc_pred = svc_classifier.predict(X_test_1)
cn = confusion_matrix(y_test ,svc_pred)
sns.heatmap(cn , annot=True)
print(confusion_matrix(y_test,svc_pred))
accuracy_score(y_test,svc_pred)

"""**random forest**"""

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=150)
RFC.fit(X_train_1,y_train)
Rfc_pred = RFC.predict(X_test_1)
print(confusion_matrix(y_test,Rfc_pred))
cn = confusion_matrix(y_test ,Rfc_pred)
sns.heatmap(cn , annot=True)

accuracy_score(y_test,Rfc_pred)

"""**pipelining with svc model**"""

from sklearn.pipeline import Pipeline

sv_classifier = SVC(kernel='rbf')
model = Pipeline([('vectorizer',cv)
                 ,('classifier',sv_classifier)])
Pipeline(steps=[('vectorizer', CountVectorizer(max_features=1420)),
                ('classifier', SVC())])

model.fit(X_train,y_train)

example_text = ["It's worst."]
example_result = model.predict(example_text)

print(example_result)

cm = confusion_matrix(y_test,model.predict(X_test))
sns.heatmap(cm,annot=True)

print(accuracy_score(y_test,model.predict(X_test)))

"""**using joblib**"""

import joblib
joblib.dump(model,'/content/classifier.pkl')

model_loaded= joblib.load('/content/classifier.pkl')

model_loaded.fit(X_train,y_train)

example_text = ["It's worst."]
example_result = model_loaded.predict(example_text)

print(example_result)

example_text = ["It's best."]
example_result = model_loaded.predict(example_text)

print(example_result)
