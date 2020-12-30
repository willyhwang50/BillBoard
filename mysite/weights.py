#Program that is used to find the optimal weights in the main task.
#Slightly different every time we run, since the Word2Vec model
#will be slightly different
#We are only roughly estimating the correct weights, and
#the difference will not be extreme

import pandas as pd
import numpy as np
import main_task as mt
import gensim
import preprocess as pre

#Load the data and then split into train and test data
data = mt.data
train = data.sample(frac=0.98, random_state=1)
test = data.drop(train.index)
train.reset_index(drop = True, inplace = True)
test.reset_index(drop = True, inplace = True)
lyrics_split = train['filtered_lyrics'].apply(lambda x: x.split())
model_train = gensim.models.Word2Vec(sentences=lyrics_split)

def find_var_equal_constant():
    '''
    Finds the constants to multiply tfidf and sentiment distances by
    '''
    n = train.shape[0]
    var_tfidf = []
    var_w2v = []
    var_sent = []

    for lyric in test['filtered_lyrics']:
        lyric_split = lyric.split()
        final_lyrics = lyrics_split.append(pd.Series([lyric_split]), 
            ignore_index=True)
        vocab_index = mt.get_vocab_index(final_lyrics)
        tfidf = mt.get_tfidf(final_lyrics, vocab_index)
        w2v = mt.get_mean_w2v(final_lyrics, model_train)
        sent = pre.find_sentiment(lyric)
        d_tfidf, d_w2v, d_sent = mt.get_distances(tfidf, w2v, sent, n, train)
        var_tfidf.append(d_tfidf.var())
        var_w2v.append(d_w2v.var())
        var_sent.append(d_sent.var())

    tfidf_constant = np.sqrt(np.mean(var_w2v) / np.mean(var_tfidf))
    sent_constant = np.sqrt(np.mean(var_w2v) / np.mean(var_sent))

    print('To equalize variance, multiply tfidf by ', tfidf_constant)
    print('To equalize variance, multiply sentiment by ', sent_constant)

def find_weights():
    '''
    Finds the optimal weights of:
    1. The three distances
    2. How similar the lyric is to a particular song vs.
       How similar tye lyric is to the artist as a whole
    '''
    d_weights = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                (1, 2, 2), (2, 1, 2), (2, 2, 1)]
    artist_song_weights = [(1, 1), (1, 4), (4, 1)]

    n_test = test.shape[0]
    weights = []
    success_percent = []

    for weight_1 in d_weights:
        a, b, c = weight_1
        for weight_2 in artist_song_weights:
            d, e = weight_2
            n_success = 0
            for i in range(n_test):
                lyric = test['lyric'][i]
                artist = mt.compute_similarity(lyric, data=train, model=model_train,
                                                weights=(a, b, c, d, e))[0]
                if test['artist'][i] in artist:
                    n_success += 1
            weights.append((a, b, c, d, e))
            success_percent.append(n_success / 40)
            print('Weights : ', (a, b, c, d, e))
            print('Percent : ', n_success / 40)

    best_percent = max(success_percent)
    best_index = success_percent.index(best_percent)
    best_weights = weights[best_index]
    
    print('The best weights is : ', best_weights)
        
    