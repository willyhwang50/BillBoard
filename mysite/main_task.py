#The file that contains the main task and the three side tasks

import pandas as pd
import numpy as np
import gensim
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import preprocess as pre

#loaded before running the tasks to save computation time
data = pd.read_csv('processed_lyrics.csv', sep='|')
model = gensim.models.Word2Vec.load('billboard.model')


def get_vocab_index(lyrics):
    '''
    Helper function for compute_similarity. Given lyrics, returns a dictionary
    with keys words, and values the index of the word

    Inputs:
        lyrics: list of list
    Returns:
        vocab_index (dict)
    '''
    total_vocab_count = {}
    for lyric in lyrics:
        for word in lyric:
            total_vocab_count[word] = total_vocab_count.get(word, 0) + 1

    final_vocab = set()
    for vocab, count in total_vocab_count.items():
        if count >= 5: #Only included words that occur at least 5 times
            final_vocab.add(vocab)
    vocab_lst = list(final_vocab)

    vocab_index = {}
    for i, vocab in enumerate(vocab_lst):
        vocab_index[vocab] = i
    
    return vocab_index

def get_tfidf(lyrics, vocab_index):
    '''
    Creates a numpy array of tfidf values, with each row
    corresponding to an artist, and each column corresponding to a word

    Inputs:
        lyrics: list of list
        vocab_index (dict)
    Returns:
        tfidf (np array)
    '''
    n = len(lyrics)
    v = len(vocab_index)
    tf = np.zeros((n, v))
    docf = np.zeros((n, v)) #document frequency

    for i in range(n):
        for word in lyrics[i]:
            if word in vocab_index:
                tf[i, vocab_index[word]] += 1
                docf[i, vocab_index[word]] = 1
        max_t = max(max(tf[i]), 1) #if max is 0, take max to be 1
        tf[i] = 0.5 + 0.5 * (tf[i] / max_t)
    docf_sum = docf.sum(axis=0)
    docf_log = np.log(n / docf_sum)
    idf = np.array(([docf_log] * n))
    tfidf = tf * idf

    #normalize
    for i in range(n):
        row = tfidf[i]
        norm = np.sqrt(np.sum(row**2))
        tfidf[i] = row / norm

    return tfidf

def get_mean_w2v(lyrics, model):
    '''
    Creates a numpy array of the mean of all word vectors
    in each lyrics

    Inputs:
        lyrics: list of list
        vocab_index (dict) 
    Returns:
        w2v (np array)
    '''
    n = len(lyrics)
    w2v = np.zeros((n, 100))

    for i in range(n):
        words = [word for word in lyrics[i] if word in model.wv.vocab] 
        if not words:
            w2v[i] = np.zeros(100)
        else:
       		mean_vec = np.mean(model.wv[words], axis=0)
        	norm = np.sqrt(np.sum(mean_vec**2))
        	w2v[i] = mean_vec/norm

    return w2v

def get_distances(tfidf, w2v, sample_sent, n, data):
    '''
    Helper function for the main task. Given tfidf, w2v, and the sentiment
    of the user input lyrics, computes the corresponding distances
    with each of the artists, and returns the corresponding numpy arrays.

    Inputs:
        tfidf (np array)
        w2v (np array)
        sample_sent (flt): sentiment of the user input lyrics
        n (int)
        data (Pandas df)
    Returns:
        a tuple of three numpy arrays
    '''
    d_tfidf = np.zeros(n)
    d_w2v = np.zeros(n)
    d_sentiment = np.zeros(n)

    for i in range(n):
        d_tfidf[i] = (1 - np.dot(tfidf[n], tfidf[i]))
        d_w2v[i] = (1 - np.dot(w2v[n], w2v[i]))
        d_sentiment[i] = abs(sample_sent - data['polarity'][i])

    return (d_tfidf, d_w2v, d_sentiment)

def compute_similarity(lyric, data=data, model=model, 
                        weights=(1, 2, 2, 4, 1)):
    '''
    The main task. Returns three most similar artists and the most similar decade
    given a user input lyric.

    Inputs:
        lyric (str)
        data (Pandas df)
        model (Word2Vec model)
        Weights: a tuple of five integers
    Returns:
        a tuple with the first value a list of three strings, and the second value
        an integer
    '''
    a, b, c, d, e = weights
    artists = list(data['artist'].unique())
    years = list(data['decade'].unique())
    n = data.shape[0]
    filtered_lyric = pre.filter_lyrics(lyric)
    lyrics = data['filtered_lyrics'].append(pd.Series(filtered_lyric),
                                            ignore_index=True)
    lyrics_split = lyrics.apply(lambda x: x.split())

    vocab_index = get_vocab_index(lyrics_split)
    tfidf = get_tfidf(lyrics_split, vocab_index)
    w2v = get_mean_w2v(lyrics_split, model)
    sample_sentiment = pre.find_sentiment(filtered_lyric)

    d_tfidf, d_w2v, d_sentiment = get_distances(tfidf, w2v,
                                            sample_sentiment, n, data)

    #distances of each song from the user input lyric
    distances = (a * d_tfidf * 10 ** 3 + b * d_w2v + 
                c * d_sentiment * 10 ** -1)

    artist_distances = np.zeros(n) #average distances of the artists
    closest_song = [] #closest songs of the artists
    for artist in artists:
        artist_i = list(data[data['artist'] == artist].index)
        d_artist = distances[artist_i]
        closest_song_index = np.argmin(d_artist) #closest song of the artist
        closest_song.append(artist_i[closest_song_index])
        avg_d_artist = np.mean(d_artist, axis=0) #average distance of the artist
        artist_distances[artist_i] = avg_d_artist

    #final distance with individual song distances and 
    #artist average distances weighted 
    final_distances = d * distances + e * artist_distances

    closest_per_artist = []
    for index in closest_song:
        closest_per_artist.append((final_distances[index], index))

    top_3 = sorted(closest_per_artist)[0:3]
    top_3_artist = []

    for pair in top_3:
        artist = data['artist'][pair[1]]
        top_3_artist.append(artist)
    
    #Most similar decade
    year_distances = np.zeros(n)
    for year in years:
        year_i = list(data[data['decade'] == year].index)
        d_year = distances[year_i]
        avg_d_year = np.mean(d_year, axis=0)
        year_distances[artist_i] = avg_d_year

    final_year_distances = d * distances + e * year_distances
    closest_year_index = np.argmin(final_year_distances)
    closest_year = data['decade'][closest_year_index]
    
    return (top_3_artist, closest_year)


def most_similar_artist(artist):
    '''
    Side Task 1. Given an artist, find the most similar artist
    within the data.

    Inputs:
        artist (str)
    Returns:
        most_similar_artist (str)
    '''

    data = pd.read_csv('most_similar.csv', sep='|')

    if artist not in data.columns:
        return 'Artist not in the data'

    return data[artist][0]

def count_per_year(term, data=data):
    '''
    Side Task 2. Given a term, plot a matplotlib plot with
    year as x, and the average count of the term per song as y.

    Inputs:
        term (str)
        data (Pandas df)
    '''
    term = term.lower()
    x = sorted(list(data['year'].unique()))
    y = []

    for year in x:
        count = 0
        year_data = data[data['year'] == year]
        lyrics_split = year_data['filtered_lyrics'].apply(lambda x: x.split())
        for lyric in lyrics_split:
            for word in lyric:
                if word == term:
                    count += 1
        y.append(count/len(year_data))
    
    plt.plot(x, y, label=term)
    plt.title('Average Word Occurence Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Count Per Song')
    plt.legend()
    plt.show()

def clear_plot():
    '''
    Clears the plot of Side Task 2
    '''
    plt.clf()

def most_positive_negative(value, data=data):
    '''
    Side Task 3. Given either an artist or a year, returns the most
    positive song and the most negative song of the particular value

    Inputs:
        value (str): artist or year
        data (Pandas df)
    Returns:
        df: Pandas dataframe with columns 'artist', 'title',
        and 'year', and rows 'Most Positive', 'Most Negative'
    '''
    artists = data['artist'].unique()
    years = [str(year) for year in data['year'].unique()]

    #Check the value is an artist in the data or a year in the data
    if value in artists:
        category = 'artist'
    elif value in years:
        category = 'year'
        value = int(value)
    else:
        return 'Arist or year not in the data'

    subset = data[data[category] == value]
    subset.reset_index(drop = True, inplace=True)
    sentiment_lst = list(subset['polarity'])
    pos = sentiment_lst.index(max(sentiment_lst))
    neg = sentiment_lst.index(min(sentiment_lst))

    df = subset.loc[[pos, neg], ['artist', 'title', 'year']]
    df.index = ['Most Positive', 'Most Negative']
    
    return df

    