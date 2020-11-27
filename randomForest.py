import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('classificationAggregate.csv')  # Import dataset

X = data.iloc[0:, :-1].values  # X values are everything except whether or not the price went up
y = data.iloc[:, 4].values  # Y value is a True/False

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Create a training dataset that is 20 percent of the overall data

sc = StandardScaler()  # Standardize the data in order to make sure training data is normally distributed
X_train = sc.fit_transform(X_train)  # Fit it with training data
X_test = sc.transform(X_test)  # Use this for the testing data

forest = RandomForestRegressor(n_estimators=500)  # Initialize the random forest algorithm with 500 estimators
forest.fit(X_train, y_train)  # Fit the model with the training data
y_pred = forest.predict(X_test)  # Create the prediction model

# Print in depth reports below including accuracy score
print(confusion_matrix(y_test, y_pred.round()))
print(classification_report(y_test, y_pred.round()))
print(accuracy_score(y_test, y_pred.round()))
