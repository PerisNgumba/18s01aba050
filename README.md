# 18s01aba050
project
import pandas as pd
df=pd.read_csv('./Tweets.csv')
df.head()

         Date	                                 tweet
0	;2015-08-03 15:35;0;0;	@ComedyCentralKE These words can kill a Luhya ...
1	;2014-07-11 23:29;0;0;	The little luhya that remains in me always ...
2	;2014-02-07 18:36;0;2;	@cheernatwildcat kill it at battle this weeken...
3	;2011-10-09 19:34;0;0;	@HomeboyzRadio H.B.R luv dat luhya hit luhyas ...
4	;2015-08-21 09:27;2;3;	#HangOutFriday hahaha ball ya terby( derby) L...

<h3>Remove empty rows and duplicates</h3>
<h4>checking for empty rows:No empty row</h4>
df.isna().sum()
Date     0
tweet    0
dtype: int64
Date     0
tweet    0
dtype: int64

df
        Date	                                tweet
0	;2015-08-03 15:35;0;0;	@ComedyCentralKE These words can kill a Luhya ...
1	;2014-07-11 23:29;0;0;	The little luhya that remains in me always ...
2	;2014-02-07 18:36;0;2;	@cheernatwildcat kill it at battle this weeken...
3	;2011-10-09 19:34;0;0;	@HomeboyzRadio H.B.R luv dat luhya hit luhyas ...
4	;2015-08-21 09:27;2;3;	#HangOutFriday hahaha ball ya terby( derby) L...
...	...	...
20015	;2012-04-17 17:20;0;0;	@symokuraya lol Yah the prices are exorbitant...
20016	;2014-02-11 06:37;0;1;	@jalangomwenyewe hao wakisii wafunguliwe radio...
20017	;2017-11-02 14:34;0;0;	Stop harbouring hatred we so hate one another...
20018	;2017-10-27 06:51;0;0;	you seem to hate kikuyus n you nickname yourse...
20019	;2017-10-17 13:48;0;0;	Why would these Kikuyus who were taught to ha...
20020 rows × 2 columns

<h4>Removing duplicate rows</h4>
df.drop_duplicates(subset ="tweet",keep = False, inplace = True)
df
<h3>Remove non-alphanumeric characters</h3>
df.tweet = df.tweet.str.replace('[^a-zA-Z0-9]', ' ')
df.head(10)

         Date	                                      tweet
0	;2015-08-03 15:35;0;0;	ComedyCentralKE These words can kill a Luhya ...
1	;2014-07-11 23:29;0;0;	The little luhya that remains in me always ...
2	;2014-02-07 18:36;0;2;	cheernatwildcat kill it at battle this weeken...
3	;2011-10-09 19:34;0;0;	HomeboyzRadio H B R luv dat luhya hit luhyas ...
4	;2015-08-21 09:27;2;3;	HangOutFriday hahaha ball ya terby derby L...
5	;2017-11-19 17:13;2;5;	Luos don t kill blood thirsty killers ar...
6	;2017-10-14 17:23;0;0;	Same police that kill luos in bondo and shot g...
7	;2017-10-14 16:34;0;0;	But there s provision for police to kill innoc...
8	;2017-10-13 21:51;0;4;	The gvnt us determined to kill all luos youn...
9	;2017-09-01 09:28;0;0;	Today events SCOK Maraga being an Adventist ...

pip install nltk
<h3>Remove stop words</h3>
import nltk

<h3>Tokenization</h3>
import nltk
from nltk.tokenize import word_tokenize
tweet = df['tweet']
word_tokens =[]
for each in tweet:
    word_tokens.append((word_tokenize(str(each))))
words_tokens = []
for each in word_tokens:
    words_tokens.append(each)
df.tweet = words_tokens
df.head()

        Date	                                tweet
0	;2015-08-03 15:35;0;0;	[ComedyCentralKE, These, words, can, kill, a, ...
1	;2014-07-11 23:29;0;0;	[The, little, luhya, that, remains, in, me, al...
2	;2014-02-07 18:36;0;2;	[cheernatwildcat, kill, it, at, battle, this, ...
3	;2011-10-09 19:34;0;0;	[HomeboyzRadio, H, B, R, luv, dat, luhya, hit,...
4	;2015-08-21 09:27;2;3;	[HangOutFriday, hahaha, ball, ya, terby, derby...

from nltk.corpus import stopwords
stopwords=nltk.corpus.stopwords.words('english')
len(stopwords)
def remove_stopwords(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean
df['tweet']=df['tweet'].apply(lambda x:remove_stopwords(x))
print(df)
<h3>Perform a Part of speech tagging</h3>
import nltk

final_tokens = []
for x in words_tokens:
    for i in x:
        final_tokens.append(i)
len(final_tokens)
tagged = nltk.pos_tag(final_tokens)
tagged[0:5]
<h3>Create a word cloud of the cleaned data</h3>
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import sys,os
os.chdir(sys.path[0])
pip install wordcloud
pip install matplotlib

pip install matplotlib
from wordcloud import WordCloud, STOPWORDS
stopwords = STOPWORDS
wordcloud = WordCloud(background_color="black", stopwords = stopwords, max_font_size=40,random_state=42).generate(str(final_tokens))
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
<h3>1.Find the number of unique words in the file</h3>
data=pd.read_csv('./Tweets.csv')
from collections import Counter
results=Counter()
data['tweet']=data['tweet'].str.lower().str.split().apply(results.update)
len(results)
39487
39487
<h3>Find the top 10 most frequent tokens in the file.</h3>
from collections import Counter
Counter(final_tokens).most_common(10)

[('are', 5369),
 ('the', 4438),
 ('to', 3195),
 ('com', 2686),
 ('is', 2681),
 ('twitter', 2562),
 ('a', 2364),
 ('you', 2344),
 ('and', 2027),
 ('of', 2003)]
[('are', 5369),
 ('the', 4438),
 ('to', 3195),
 ('com', 2686),
 ('is', 2681),
 ('twitter', 2562),
 ('a', 2364),
 ('you', 2344),
 ('and', 2027),
