# B365 Final Course Project — TSLA Stock Price Predictor
Tested in Python 3.8

#### Easy install all required libraries:
`pip install watson-developer-cloud pandas scikit-learn matplotlib numpy yfinance`

#### Individual Libraries Required:
1. [Watson Developer Cloud](https://pypi.org/project/watson-developer-cloud/)
2. [Pandas](https://pypi.org/project/pandas/)
3. [Scikit-Learn](https://pypi.org/project/scikit-learn/)
4. [Matplotlib](https://pypi.org/project/matplotlib/)
5. [Numpy](https://pypi.org/project/numpy/)
6. [Yahoo Finance](https://pypi.org/project/yfinance/)

#### List of possible algorithms:
1. Decision Trees in `decisionTree.py`
2. k-Nearest Neighbour in `knn.py`
3. Naive Bayes in `naiveBayes.py`
4. Random Forest in `randomForest.py`
5. Linear Regression in `regression.py`
6. Neural Network in `neuralNetwork.py`

Simply run any of the Python files and keep all the `csv` and `json` files in the same directory to ensure proper functionality

## How to Generate the Dataset
#### Tweet Dataset
In order to generate the tweet dataset, you must install [Twint](https://github.com/twintproject/twint)
This is a Python library that can bypass the official Twitter API limitations on downloading Tweets
Simply run `pip3 install twint` to install
Then, in your command line run two commands:
1. `twint -u elonmusk --since 2010-06-29 -o tweets.csv --csv`
2. `twint -u elonmusk --since 2010-06-29 -o tweets.json --json`

These two commands will generate the Tweet dataset in `csv` and `json` formats

Next, run the `cleanTweet.py` program in order to remove attributes that are not necessary and create sub data sets that will make it easier to aggregate in the main file

#### Historical Stock Data
Next, run the `historicalData.py` in order to create a `csv` with TSLA historical values
Run again and edit the `tickerName` variable to `QQQ` in order to generate NASAQ historical values and change this line:

`df.to_csv(r"historicals.csv", index=True, header=True)` to this: `df.to_csv(r"nasdaq.csv", index=True, header=True)`

This will generate the TSLA and NASDAQ financial data sets

#### Putting it all together
Simply run the `main.py` file in order to generate the `aggregate.csv` dataset that is used for regression

Then, run the `classificationDataset.py` file in order to create the `classificationAggregate.csv` dataset that is used for classification
