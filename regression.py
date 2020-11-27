import pandas
from sklearn import linear_model
from matplotlib import pyplot as plt

df = pandas.read_csv("aggregate.csv")  # Import dataset

# Sole regression model that we implemented is in this code
X = df[['sentiment', 'tweetCount']]  # Independent variables fed in are sentiment and tweet count
y = df['close']  # Dependent variable is the closing price

regression = linear_model.LinearRegression()  # Initialize a regression model
regression.fit(X, y)  # Input the x and y

print(regression.coef_)  # Regression coefficients

x1 = df['date']

plt.scatter(x1, y, color='g')  # Plot date versus the stock price

plt.plot(x1, regression.predict(X), color='k')  # Plot date versus the model
plt.show()  # Render

# This model lacked any useful data and showed us that we cannot use regression as a valid method of
# predicting the market price. Instead we decided to use classification
