import yfinance as yf  # Use yfinance module to gather financial data
import pandas as pd


def getHistory(ticker, startDate, endDate):
    # Use yfinance to get data and put it into a data frame
    data = yf.download(str(ticker), start=str(startDate), end=str(endDate), group_by='column')
    df = pd.DataFrame(data)
    # Remove two useless columns
    df.pop("Adj Close")
    df.pop("Volume")
    # Export to a CSV file
    df.to_csv(r"historicals.csv", index=True, header=True)
    print("Historical data operation successful!")


# Enter details of TSLA for the function
tickerName = "TSLA"
startDate = "2011-12-01"
endDate = "2020-11-01"

getHistory(tickerName, startDate, endDate)
