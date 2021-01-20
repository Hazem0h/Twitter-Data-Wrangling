# Twitter-Data-Wrangling
This is a project in the Udacity Data professional Nanodegree, sponsored by fwd. In this project, I gathered data from different sources, assessed the data, and cleaned it. In the end, I extracted some insights from that clean data, with visualizations.


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

# 1) Gathering Data

## 1.1 Downloadable csv file


```python
twitter_archive_df = pd.read_csv("twitter-archive-enhanced.csv")
twitter_archive_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892420643555336193</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 16:23:56 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Phineas. He's a mystical boy. Only eve...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892420643...</td>
      <td>13</td>
      <td>10</td>
      <td>Phineas</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 00:17:27 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Tilly. She's just checking pup on you....</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892177421...</td>
      <td>13</td>
      <td>10</td>
      <td>Tilly</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-31 00:18:03 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Archie. He is a rare Norwegian Pouncin...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891815181...</td>
      <td>12</td>
      <td>10</td>
      <td>Archie</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>891689557279858688</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-30 15:58:51 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Darla. She commenced a snooze mid meal...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891689557...</td>
      <td>13</td>
      <td>10</td>
      <td>Darla</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>891327558926688256</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-29 16:00:24 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Franklin. He would like you to stop ca...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891327558...</td>
      <td>12</td>
      <td>10</td>
      <td>Franklin</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## 1.2 The image prediciton data via requesting url

### For this part, we will be using the *requests* library
In order to parse the request succesfully, we will need to take care of 3 things:
- the request status code "e.g: 200 for success, 404 for failure"
- the request encoding "like utf-8"
- the content type "JSON, or csv, or tsv"

#### 1.2.1 fetching the request data


```python
# importing the library
import requests
```


```python
# fetching the request object
url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
req = requests.get(url)
```


```python
# asserting a successful response code
assert(req.status_code == 200)
```

#### 1.2.2 content type and encoding

The encoding of the response, based on which we will decode it


```python
encoding = req.encoding
encoding
```




    'utf-8'



Content Type


```python
req.headers["content-type"]
```




    'text/tab-separated-values; charset=utf-8'



So the content type is, as we expect, a tsv file format

#### 1.2.3 Reading the data into a dataframe
In order to decode the data into a dataframe, we will need 2 steps
- specify the encoding scheme
- specify the delimiter in the pandas read_csv function as "\t" 

<br>I had to use "io" library, I don't know exactly why, but without it, the code crashes. <br>
[Here](https://stackoverflow.com/questions/39213597/convert-text-data-from-requests-object-to-dataframe-with-pandas) is the stack overflow post I referred to.


```python
import io
```


```python
image_prediction_df = pd.read_csv(io.StringIO(req.content.decode(encoding)), sep = "\t")
```


```python
image_prediction_df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>jpg_url</th>
      <th>img_num</th>
      <th>p1</th>
      <th>p1_conf</th>
      <th>p1_dog</th>
      <th>p2</th>
      <th>p2_conf</th>
      <th>p2_dog</th>
      <th>p3</th>
      <th>p3_conf</th>
      <th>p3_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>667369227918143488</td>
      <td>https://pbs.twimg.com/media/CUL4xR9UkAEdlJ6.jpg</td>
      <td>1</td>
      <td>teddy</td>
      <td>0.709545</td>
      <td>False</td>
      <td>bath_towel</td>
      <td>0.127285</td>
      <td>False</td>
      <td>Christmas_stocking</td>
      <td>0.028568</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1643</th>
      <td>808001312164028416</td>
      <td>https://pbs.twimg.com/media/CzaY5UdUoAAC91S.jpg</td>
      <td>1</td>
      <td>Labrador_retriever</td>
      <td>0.730959</td>
      <td>True</td>
      <td>Staffordshire_bullterrier</td>
      <td>0.130726</td>
      <td>True</td>
      <td>American_Staffordshire_terrier</td>
      <td>0.028853</td>
      <td>True</td>
    </tr>
    <tr>
      <th>475</th>
      <td>675149409102012420</td>
      <td>https://pbs.twimg.com/media/CV6czeEWEAEdChp.jpg</td>
      <td>1</td>
      <td>chow</td>
      <td>0.999876</td>
      <td>True</td>
      <td>Tibetan_mastiff</td>
      <td>0.000059</td>
      <td>True</td>
      <td>Tibetan_terrier</td>
      <td>0.000029</td>
      <td>True</td>
    </tr>
    <tr>
      <th>289</th>
      <td>671163268581498880</td>
      <td>https://pbs.twimg.com/media/CVBzbWsWsAEyNMA.jpg</td>
      <td>1</td>
      <td>African_hunting_dog</td>
      <td>0.733025</td>
      <td>False</td>
      <td>plow</td>
      <td>0.119377</td>
      <td>False</td>
      <td>Scottish_deerhound</td>
      <td>0.026983</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1352</th>
      <td>759923798737051648</td>
      <td>https://pbs.twimg.com/media/CovKqSYVIAAUbUW.jpg</td>
      <td>1</td>
      <td>Labrador_retriever</td>
      <td>0.324579</td>
      <td>True</td>
      <td>seat_belt</td>
      <td>0.109168</td>
      <td>False</td>
      <td>pug</td>
      <td>0.102466</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## 1.3 The data collected via Twitter API and tweepy library
The first step is to query for the tweets whose ids exist in the twitter_archive_enhanced.csv file<br>
This is complete and all JSON data was saved in `tweets_json.txt`

The following illustrate why we did so:
- We want 2 extra features which are not available in the current datasets: which are number of likes and number or retweets
- As per the project description, The ratings probably aren't all correct. Same goes for the dog names and probably dog stages, so string matching is in order
- As such We are only interested in the tweets whose ids belong to the current dataset

#### Important *note*
While requesting data via twitter API, some tweets didn't load, probably becuause they were probably deleted<br>
The difference is about `25 tweets` so dropping these tweets will not probably affect the analysis. <br>
We will go through this again in the assessment

### 1.3.1 Parsing the data from the `tweets_json.txt` file 
for this task, we will json library to facilitate parsing json 


```python
import json
id_list = []
faves_list = []
retweets_list = []
text_list = []

# opening the tweets_json file
with open("tweets_json.txt", "r") as file:
    for line in file:
        tweet_data = json.loads(line)
        id_list.append(tweet_data["id"])
        faves_list.append(tweet_data["favorite_count"])
        retweets_list.append(tweet_data["retweet_count"])
        text_list.append(tweet_data["text"])
```

### 1.3.2 Creating a Dataframe from the parsed data 


```python
API_dict = {"tweet_id": id_list, "text": text_list, "favorite_count":faves_list, "retweet_count":retweets_list}
API_df = pd.DataFrame(API_dict)
API_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>text</th>
      <th>favorite_count</th>
      <th>retweet_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892420643555336193</td>
      <td>This is Phineas. He's a mystical boy. Only eve...</td>
      <td>35527</td>
      <td>7511</td>
    </tr>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>This is Tilly. She's just checking pup on you....</td>
      <td>30750</td>
      <td>5575</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>This is Archie. He is a rare Norwegian Pouncin...</td>
      <td>23115</td>
      <td>3694</td>
    </tr>
    <tr>
      <th>3</th>
      <td>891689557279858688</td>
      <td>This is Darla. She commenced a snooze mid meal...</td>
      <td>38816</td>
      <td>7695</td>
    </tr>
    <tr>
      <th>4</th>
      <td>891327558926688256</td>
      <td>This is Franklin. He would like you to stop ca...</td>
      <td>37095</td>
      <td>8295</td>
    </tr>
  </tbody>
</table>
</div>



And that concludes our data gathering phase<br>
We collected data from 3 different sources

# 2) Assessment

### Now it is time to assess data both viusally and programmatically to see if there are issues to be fixed

## `issue `: twitter archive dataset has "None" string instead of Nan for null values in doggo, floofer, pupper, puppo tables
#### these columns must have null values instead the string "None". To prove this is an issue, run the following cells


```python
twitter_archive_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892420643555336193</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 16:23:56 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Phineas. He's a mystical boy. Only eve...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892420643...</td>
      <td>13</td>
      <td>10</td>
      <td>Phineas</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 00:17:27 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Tilly. She's just checking pup on you....</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892177421...</td>
      <td>13</td>
      <td>10</td>
      <td>Tilly</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-31 00:18:03 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Archie. He is a rare Norwegian Pouncin...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891815181...</td>
      <td>12</td>
      <td>10</td>
      <td>Archie</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>891689557279858688</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-30 15:58:51 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Darla. She commenced a snooze mid meal...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891689557...</td>
      <td>13</td>
      <td>10</td>
      <td>Darla</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>891327558926688256</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-29 16:00:24 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Franklin. He would like you to stop ca...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891327558...</td>
      <td>12</td>
      <td>10</td>
      <td>Franklin</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
first_doggo = twitter_archive_df.doggo[0]
print("The first element in 'doggo' column is '{}' and its type is {}".format(first_doggo, type(first_doggo)))
```

    The first element in 'doggo' column is 'None' and its type is <class 'str'>
    

So apparently, this needs to be fixed, becuase then the isnull() method won't identify the string "None" as null


```python
twitter_archive_df.doggo.isnull()[0]
```




    False



## `issue ` twitter archive dataset columns doggo, floofer, pupper, puppo are values rather than variable names
#### These columns need to be grouped under one column like "dog_stage"


```python
twitter_archive_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>timestamp</th>
      <th>source</th>
      <th>text</th>
      <th>retweeted_status_id</th>
      <th>retweeted_status_user_id</th>
      <th>retweeted_status_timestamp</th>
      <th>expanded_urls</th>
      <th>rating_numerator</th>
      <th>rating_denominator</th>
      <th>name</th>
      <th>doggo</th>
      <th>floofer</th>
      <th>pupper</th>
      <th>puppo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892420643555336193</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 16:23:56 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Phineas. He's a mystical boy. Only eve...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892420643...</td>
      <td>13</td>
      <td>10</td>
      <td>Phineas</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>892177421306343426</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-08-01 00:17:27 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Tilly. She's just checking pup on you....</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/892177421...</td>
      <td>13</td>
      <td>10</td>
      <td>Tilly</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>891815181378084864</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-31 00:18:03 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Archie. He is a rare Norwegian Pouncin...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891815181...</td>
      <td>12</td>
      <td>10</td>
      <td>Archie</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>891689557279858688</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-30 15:58:51 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Darla. She commenced a snooze mid meal...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891689557...</td>
      <td>13</td>
      <td>10</td>
      <td>Darla</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>891327558926688256</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-07-29 16:00:24 +0000</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>This is Franklin. He would like you to stop ca...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://twitter.com/dog_rates/status/891327558...</td>
      <td>12</td>
      <td>10</td>
      <td>Franklin</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## `issue` in twitter archive dataset The "rating_denominator" column is redundant
This column has the same value for all tweets, so it has no added value to be included in the dataset <br>
Instead, we can refer to the denominator of the rating in a documentation for the dataset, but not in the dataset itself<br>
Denminators not equalt to 10 are probably errors, rather than different values of the denominator


```python
twitter_archive_df.rating_denominator.value_counts()
```




    10     2333
    11        3
    50        3
    80        2
    20        2
    2         1
    16        1
    40        1
    70        1
    15        1
    90        1
    110       1
    120       1
    130       1
    150       1
    170       1
    7         1
    0         1
    Name: rating_denominator, dtype: int64



## `issue ` in twitter archive dataset, incorrect data in rating_denominator, and rating_numerator
As explained by the person who collected the data, some data may be erroneuous, and as explianed above, there are some rating denominators that are clearly wrong

## `issue` in twitter archive dataset, incorrect data in dog_name and stage name
This is because the data collector said "The ratings probably aren't all correct. Same goes for the dog names and probably dog stages too" <br>
For example, the dog name is not always capitalized


```python
twitter_archive_df.name.str.istitle().all()
```




    False



## `issue` twitter archive dataset: dog names are sometimes upper case and sometimes lower case


```python
# this code checks if all dog names are capitalized
twitter_archive_df.name.str.istitle().all()
```




    False



## `issue` incorrect datatypes in twitter archive dataset


```python
twitter_archive_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2356 entries, 0 to 2355
    Data columns (total 17 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   tweet_id                    2356 non-null   int64  
     1   in_reply_to_status_id       78 non-null     float64
     2   in_reply_to_user_id         78 non-null     float64
     3   timestamp                   2356 non-null   object 
     4   source                      2356 non-null   object 
     5   text                        2356 non-null   object 
     6   retweeted_status_id         181 non-null    float64
     7   retweeted_status_user_id    181 non-null    float64
     8   retweeted_status_timestamp  181 non-null    object 
     9   expanded_urls               2297 non-null   object 
     10  rating_numerator            2356 non-null   int64  
     11  rating_denominator          2356 non-null   int64  
     12  name                        2356 non-null   object 
     13  doggo                       2356 non-null   object 
     14  floofer                     2356 non-null   object 
     15  pupper                      2356 non-null   object 
     16  puppo                       2356 non-null   object 
    dtypes: float64(4), int64(3), object(10)
    memory usage: 313.0+ KB
    

The following datatypes are not the best suitable for data manipulation
- `timestamp` should be datetime
- `in_reply_to_status_id` should be int or string
- `in_reply_to_user_id` should be int or string
- `retweeted_status_id` should be int or string
- `retweeted_status_user_id` should be int or string
- `retweeted_status_timestamp` should be datetime

## `issue` missing data from the image prediction dataset

The number of observations in the image predcition dataset is smaller than that in the twitter archive dataset


```python
print("Archive dataset dimensions {} \n wheras image prediction dimensions are {}".format(twitter_archive_df.shape, image_prediction_df.shape))
```

    Archive dataset dimensions (2356, 17) 
     wheras image prediction dimensions are (2075, 12)
    

So there are 2356 rows in the archive dataset <br>
while there are 2075 rows in the image prediction dataset

## `issue` missing data from the twitter API dataset
In the twitter API dataset, some tweets failed to load, probably because they were deleted


```python
print(twitter_archive_df.shape)
print(API_df.shape)
```

    (2356, 17)
    (2331, 4)
    

So it is clear that the API data has 25 less tweets

## `issue` ambiguous column name in the image prediction dataset
The columns `p1`, `p2`, `p3` and `p1_conf`, `p2_conf`... can be ambiguous, and should be named like `prediction_1` to give better semantic


```python
image_prediction_df.columns
```




    Index(['tweet_id', 'jpg_url', 'img_num', 'p1', 'p1_conf', 'p1_dog', 'p2',
           'p2_conf', 'p2_dog', 'p3', 'p3_conf', 'p3_dog'],
          dtype='object')



## `issue` Data tables are not properly grouped
This is a tidiness issue. The tweet information exists in 2 tables: archive dataset and twitter api dataset<br>
The dog information exists in 2 tables: the archive dataset and the image prediction dataset

## To summarize the issues:
### Tidiness issues:
- doggo, floofer, pupper, puppo are varaible values instead of distinct variable names `archive dataset`
- Data tables not properly grouped `all datasets`
- rating denominator is redundant `archive dataset`, and not considered a variable

### Quality issues:
- using the string "None" instead of Nan for empty values in doggo, floofer, pupper, puppo columns `archive dataset`
- errors in collecting data regarding the rating_numerator and rating denominator in `archive dataset`
- possible incorrect parsing of tweet text for dog name and dog stage name `archive dataset`
- dog names are inconsistently upper and lower case `archive dataset`
- improper datatypes in the `archive dataset`
- missing data from the `image prediciton dataset`
- missing data from the `twitter APi dataset`
- ambiguous column names in `image prediciton dataset`

# 3) Cleaning
First thing to do, is to create copies of the dataframes


```python
twitter_archive_clean = twitter_archive_df.copy()
image_predictio_clean = image_prediction_df.copy()
api_clean = API_df.copy()
```

### `Resolve` missing data from twitter API dataset

#### Define
will remove tweets in the archive dataset that do not exist in the API dataset


#### Code


```python
twitter_archive_clean = twitter_archive_clean.query("tweet_id in @api_clean.tweet_id.values")
```

#### Test
We will test that the dataframes archive dataset and api dataset have the same tweet ids


```python
assert(twitter_archive_clean.shape[0] == api_clean.shape[0])
for tweet_id in twitter_archive_clean.tweet_id.values:
    assert(tweet_id in api_clean.tweet_id.values)
```

### `Resolve`: incorrect dog names and/or dog stage names

#### Define
We will be using the api dataset to check for correctness of parsing this data<br>
We could have reparsed the tweet text in the archive dataset, but this is to leave no chance to error, since the archive dataframe has many flaws


```python
with open ("tweets_json.txt", "r") as file:
    i = 0
    for line in file:
        tweet = json.loads(line)
        if tweet["retweeted"] == False:
            
    print(i)
```

    2331
    


```python

```
