import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split# Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#from sklearn import StringIO  
#from sklearn.externals.six import StringIO
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

col_names = ['InfectionPercentage', 'DeathPercentage', 'Weather', 'SchoolOpen']
pi = pd.read_csv("DecissiontreeSuperPerform.csv", header=None, names=col_names)
#pi=pi.drop('Unnamed: 1',axis='columns') #drop Unnecessary Column
#pi=pi.drop('Unnamed: 3',axis='columns') #drop Unnecessary Column

#split dataset in features and target variable
feature_cols = ['InfectionPercentage', 'DeathPercentage', 'Weather']
X = pi[feature_cols] # Features
y = pi.SchoolOpen # Target variable


# Split dataset into training set and test set
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
#= train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

clf
#Predict the response for test dataset
X_train
X_test
X_train
y_train
y_test
X_test
X_train
clf = clf.fit(X_train,y_train)
clf
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_pred, y_test))

print("Accuracy:",metrics.accuracy_score(y_pred, y_test))
y_train
X_train

y_pred

y_test

X_test
y_pred = clf.predict(X_train)
print("Accuracy:",metrics.accuracy_score(y_train, y_pred))
y_pred
y_train
X_train
X_train
y_train
print("Accuracy:",metrics.accuracy_score(y_train, y_pred))







