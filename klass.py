
#!/usr/bin/env python
# coding: utf-8

import pandas as pd;
import matplotlib.pyplot as plt

genes96 = pd.read_csv('C:/temp/096_Frozen_BMMCs(Healthy_Control1)_csv.csv')
genes97= pd.read_csv('C:/temp/097_Frozen_BMMCs(Healthy_Control2)_csv.csv')
t96= genes96.transpose()
t96.shape
t96.columns=t96.iloc[0]
t96a = t96.iloc[1:]
t96a.shape
t96a =  t96a.assign(result='Healthy_Control1')
t96a.shape
t97= genes97.transpose()
t97.columns=t97.iloc[0]
t97a = t97.iloc[1:]
t97a.shape
t97a =  t97a.assign(result='Healthy_Control2')
t97a.shape
final= pd.concat([t96a,t97a])
final.shape


final.to_csv('panofinal.csv', sep=',', encoding='utf-8')


from sklearn.model_selection import train_test_split
X = final.loc[:, final.columns != 'result']
y = final.loc[:,final.columns == 'result']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))




from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                             random_state=1)
clf.fit(X_train, y_train)

print('Accuracy of Random Forest  classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
