# CS7641: Assignment 1
# Algorithm: Decision Tree

from sklearn import tree
X = [[0,0], [1,1]]
Y = [0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
clf.predict([[2., 2.]])
