import tweepy
import numpy as np
import pandas as pd
import json

# twitter API stuff
consumer_key = 'xxxxxxx'
consumer_secret = 'xxxxxxx'
access_token = "xxxxxxx"
access_secret = 'xxxxxx'

# authenticating
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# importing the dataframe twitter-archive-enhanced
twitter_archive = pd.read_csv("twitter-archive-enhanced.csv")
twitter_archive.head()

# getting the tweets in the archive via the tweepy api
failed_ids = []
with open ("tweet_json.txt", "w") as file:
    for tweet_id in twitter_archive.tweet_id.values:
        try:
            tweet = api.get_status(tweet_id)
        except:
            print("failed to get tweet with id {}".format(tweet_id))
            failed_ids.append(tweet_id)
            continue
        json.dump(tweet._json, file)
        file.write("\n")
        print("written tweet with ID {}".format(tweet._json["id"]))