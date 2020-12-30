#Filters artist_lyrics.txt and turns them into processed_lyrics.csv,
#in order to be used in main_task.py. Also creates billboard.model.

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import gensim

data = pd.read_csv('artist_lyrics.txt', sep='|')
sid = SentimentIntensityAnalyzer()

def filter_lyrics(lyric):
    '''
    Filters the lyrics in artist_lyrics.txt.
    Turns each word into lowercase, strips punctuation.
    Note: Did not remove stopwords because they can carry significance
    in terms of the artist's unique style

    Inputs:
        lyric (str)
    Returns:
        filtered lyric (str)
    '''
    words_tokens = lyric.split()
    filtered_words = []

    for word in words_tokens:
        word = word.lower().strip(string.punctuation)
        filtered_words.append(word)

    return ' '.join(filtered_words)

def create_filtered_lyrics_col(df):
    '''
    Creates 'filtered_lyrics' column given a dataframe with lyrics

    Inputs:
        df (Pandas dataframe)
    '''
    df['filtered_lyrics'] = df['lyric'].apply(filter_lyrics)

def find_sentiment(lyric):
    '''
    Finds the sentiment (how negative or positive) of the given lyric

    Inputs:
        lyric (str)
    Returns:
        sentiment (flt): float from -1 to 1
    '''

    return sid.polarity_scores(lyric)['compound']

def create_polarity_col(df):
    '''
    Creates 'polarity' column given a dataframe with lyrics

    Inputs:
        df (Pandas dataframe)
    '''
    df['polarity'] = df['filtered_lyrics'].apply(find_sentiment)

def create_decade_col(df):
    '''
    Creates 'decade' column given a datafrmae with lyrics

    Inputs:
        df (Pandas dataframe)
    '''
    df['decade'] = None
    decades = [1950, 1960, 1970, 1980, 1990, 2000, 2010]

    for decade in decades:
        df_bool = (decade <= data['year']) & (data['year'] < decade + 10)
        df.loc[df_bool, 'decade'] = str(decade)

def create_model(df):
    '''
    Creates Word2Vec model using words in the lyrics given a dataframe
    with lyrics

    Inputs:
        df (Pandas dataframe)
    Returns:
        model (Word2Vec model)
    '''
    lyrics_split = data['filtered_lyrics'].apply(lambda x: x.split())
    model = gensim.models.Word2Vec(sentences=lyrics_split, size=100, window=5)
    
    return model


create_filtered_lyrics_col(data)
create_polarity_col(data)
create_decade_col(data)
data.to_csv('processed_lyrics.csv', index=False, sep='|')
model = create_model(data)
model.save('billboard.model')


