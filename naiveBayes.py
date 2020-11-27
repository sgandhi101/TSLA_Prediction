import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from numpy import median
import matplotlib.pyplot as plt

data = pd.read_csv('classificationAggregate.csv')  # Import dataset


# Function that creates the Bayes classifier
def bayes():
    X = data.iloc[0:, :-1].values  # X values are everything except whether or not the price went up
    y = data.iloc[:, 4].values  # Y value is a True/False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # Create a training dataset that is
    # 20 percent of the overall data

    model = GaussianNB()  # Initialize a Naive Bayes Classifier
    model.fit(X_train, y_train)  # Fit it with the X and Y inputs specified above

    y_pred = model.predict(X_test)  # Create a model

    accuracy = accuracy_score(y_test, y_pred) * 100  # Determine the accuracy score as a percentage
    return accuracy


values = []  # List of accuracy
for i in range(1000):
    values.append(bayes())  # Iterate through the classifier a 1000 times and append accuracy
    print("Iteration", i, "Complete")

print("Median accuracy:", median(values), "%")  # Median accuracy of the Naive Bayes Classifier
