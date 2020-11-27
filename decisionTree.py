import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("classificationAggregate.csv")  # Import dataset

X = dataset.drop('positiveChange', axis=1)  # X values are everything except whether or not the price went up
y = dataset['positiveChange']  # Y value is a True/False


# Create a function for the decision tree
def decisionTree():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # Create a training dataset that is 20 percent of the overall data
    classifier = DecisionTreeClassifier()  # Initialize the decision tree
    classifier.fit(X_train, y_train)  # Fit the training data
    y_pred = classifier.predict(X_test)  # Use this to predict
    return (accuracy_score(y_test, y_pred))  # Return the accuracy of this particular model


accuracy = []
for i in range(1000):
    accuracy.append(decisionTree())  # Iterate through the classifier a 1000 times and append accuracy
    print("Iteration", i, "Complete")

print("Median Accuracy:", numpy.median(accuracy) * 100, "%")
# Median accuracy of a decision tree with 1000 iterations
