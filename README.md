### This is a project in the Udacity Data professional Nanodegree, sponsored by fwd.

In this project, I was required to perform data wrnagling in 3 main steps:
1- Data gathering
2- Data Assessing
3- Data Cleaning

The data is about rating dogs, based on the "sarcastic" twitter account "[https://twitter.com/dog_rates?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor] weRateDogs". Each data entry contains information about the dog's name, breed, rating (out of 10) as well as information about the tweet itself, like the retweet count and favorite count.

In the first step, Data gathering, I collected data from 3 different sources
- A downloadable csv file "twitter-archive-enhanced.csv"
- A file that is requested (via the "requests" library) "image_prediction.tsv"
- Twitter API "tweet_json.txt"

Where the code in which I used the Twitter API is in the file (API_data.py) and I obscured my own credentials. For this to run, you will need to have a twitter developer account, and use your own credentials.
I saved the data in the file "tweet_json.txt" where the data is stored in JSON format. I used the "json" library to parse the file.

In the second step, Assessing data, I identified several cleanliness and organizational issues with the data that need to be fixed.

And naturally in the third step, I performed the required data cleaning. The cleaned data is stored in the file "twitter_archive_master.csv"

After these 3 main steps, I was required to provide some insights from the clean data, which I communicated through graphs and statistical means. You can find these insights at the end of the notebook or in the "insights_report.pdf"
