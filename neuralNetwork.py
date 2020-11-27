# DO *NOT* RUN ON LOCAL COMPUTER, SEE NOTE BELOW BEFORE ATTEMPTING

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('classificationAggregate.csv')  # Import the dataset

X = data.iloc[0:, :-1]  # X values are everything except whether or not the price went up
y = data.iloc[:, 4]  # Y value is a True/False

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Create a training dataset that is 10 percent of the overall data
# The reason that this one is lower is that we found that a neural network over fits the data much more easily
# than any of the other classifiers we trained so we found that 10% is better for our results

scale = StandardScaler()  # Standardize the data in order to make sure training data is normally distributed
scale.fit(X_train)  # Fit it with training data

# Apply this to the rest of the data
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

# NOTE: WE STRONGLY RECOMMEND YOU DO *NOT* RUN THIS ON YOUR LOCAL COMPUTER. WE RAN THIS ON IU'S CARBONATE
# SUPERCOMPUTER AND IT STILL TOOK A CONSIDERABLE AMOUNT OF TIME TO RUN. IT WILL MOST LIKELY CRASH YOUR
# LOCAL COMPUTER AS IT REQUIRES *MUCH* MORE RESOURCES THAN ANY OF THE OTHER ALGORITHMS

# Train a neural network with six nodes with 5000 hidden layers each iterated through 10 million times
# These numbers changed a lot we played around with different numbers of nodes, layers, and iterations
mlp = MLPClassifier(hidden_layer_sizes=(5000, 5000, 5000, 5000, 5000, 5000), max_iter=10000000)
mlp.fit(X_train, y_train.values.ravel())  # Fit the training data

predictions = mlp.predict(X_test)  # Create the model

# Print a list of accuracy metrics
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
