import pandas as pd
from sentiment import sentiment

tweets = pd.read_csv("cleaned_twitter_data.csv")  # Import cleaned Twitter data

# Iterate through every tweet and get a sentiment rating through the
# sentiment function
tweet_sentiment_rating = []
for index, row in tweets.iterrows():
    try:
        tweet_sentiment_rating.append(sentiment(row['tweet']))
    except:
        # In the event that the tweet is too short, formatted strangely, or in another language
        # It can get skipped
        print("LINE " + str(index) + " SKIPPED DUE TO FORMATTING ERRORS: " + str(row['tweet']))
    # Put all the sentiment ratings in the dataframe in a new column
    tweets['Sentiment'] = pd.Series(tweet_sentiment_rating)
# Export to a new file
tweets.to_csv(r"sentimentAnalysis.csv", index=True, header=True)
# Import stock data and sentiment analysis
df1 = pd.read_csv("historicals.csv")
df2 = pd.read_csv("sentimentAnalysis.csv")

# Removes 0 values that signify that the algorithm was unable to find an associated sentiment
df2 = df2[df2.Sentiment != 0]

# Take the mean sentiment of every day to make it easy to compare to stock data
df2 = df2.groupby('date', as_index=False).mean()

# Merges based on date
aggregate = pd.merge(df1, df2, on='date')
aggregate.to_csv(r"unfinishedAggregate.csv", index=False, header=True)

# Random column generated as a result of merging that is wiped
cleanAggregate = pd.read_csv(r"unfinishedAggregate.csv")
del cleanAggregate['Unnamed: 0']
cleanAggregate.to_csv(r"unfinishedAggregate.csv", index=False, header=True)

# Import the unfinished dataset from above and the media data
df1 = pd.read_csv("unfinishedAggregate.csv")
df2 = pd.read_csv("pictureData.csv")

# Merge the two together based on date to add media data
aggregate1 = pd.merge(df1, df2, on='date')

# Add the number of tweets per day as a column
number = pd.read_csv("tweetsPerDay.csv")
aggregate2 = pd.merge(aggregate1, number, on='date')

# Produce the final file
aggregate2.to_csv(r"aggregate.csv", index=False, header=True)
