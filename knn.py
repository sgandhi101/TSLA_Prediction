import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('classificationAggregate.csv')  # Import dataset

X = data.iloc[0:, :-1].values  # X values are everything except whether or not the price went up
y = data.iloc[:, 4].values  # Y value is a True/False

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Implement feature scaling
scaler = StandardScaler()  # Standardize the data in order to make sure training data is normally distributed
scaler.fit(X_train)  # Fit it with training data
# Use this for the testing data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Initialize a test case of the K nearest neighbours with 5 Neighbours
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)  # Fit the model with the training data

y_pred = classifier.predict(X_test)  # Use this model to predict the rest of the dataset

# Print out a plethora of results including the accuracy
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# NOW, we apply this same concept except in mass with K values between 1 and 600
error = []
lowestError = 1  # Start off with the worst possible error and this will get compared every iteration to find the lowest
for i in range(1, 600):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    x = np.mean(pred_i != y_test)
    if x < lowestError:
        lowestError = x

# Print out all our error rate data to determine the accuracy of the model
print("Lowest Error Rate", lowestError * 100, "%")
print("Median Error Rate", np.median(error) * 100, "%")
print("Median Accuracy Rate:", 100 - (np.median(error) * 100), "%")

# Output a graph with matplotlib in order to see our results graphed
plt.figure(figsize=(12, 6))
plt.plot(range(1, 600), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
