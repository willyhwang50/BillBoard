# CS 122 Project: Billboard Analysis
# Raw data processing & Lyrics crawling
#
# Last updated March 7 2020

import bs4
import urllib3
import certifi
import unidecode
import pandas as pd
import re


punctuations = [".", ",", "?", "!", "'", ";", ":",'"',"/", "-", "_", "(", ")", 
                "*", "[", "]", "&", "%", "@", "`", "~", "$", "^", "+", "=", 
                "|", "#"]
title_edit = ["/", "("]
singer_edit = ["Feat", "& ", ", ", "/", "("]
guide = {0: ("p", "songLyricsV14 iComment-text", "-"), 
         1: ("div", "lcontent", ""), 2: ("p", "lyric-text", "-"),
         3: ("div", "lyrics", "-")}
filters = ["sbmtl", "Video:", "Sorry, we have no", "We do not have the lyrics"]


class artist:
    
    def __init__(self, name):
        '''
        Constructor for the artist class
        '''
        self.name = name
        self.songs = []
        

class song:
    
    def __init__(self, artist, title, lyric, year):
        '''
        Constructor for the song class
        '''
        self.artist = artist
        self.title = title
        self.lyric = lyric
        self.year = year


def clean(dataframe):
    '''
    Processes artist name and song title data into an appropriate format for
    crawling and analysis.
    
    Input: dataframe (pandas df) of original Billboard data
    Returns: df with Song and Performer columns processed
    '''
    cleanname, cleantitle = [], []
    
    for i in range(len(dataframe)):
        
        singer = unidecode.unidecode(dataframe["Performer"][i])
        title = unidecode.unidecode(dataframe["Song"][i])

        for symb in title_edit:
            if symb in title:
                title = title[:title.index(symb)]
                break
  
        for symb in singer_edit:
            if symb in singer:
                singer = singer[:singer.index(symb)]
                break

        for symb in punctuations:
            singer = singer.replace(symb, "")
            title = title.replace(symb, "")
        
        #manual handling of few irregular cases
        if singer == "Pnk":
            singer = "P!nk"
        
        if singer == "Elvis Presley With The Jordanaires":
            singer = "Elvis Presley"

        if singer != "" and singer[-1] == " ":
            singer = singer[:-1]
        
        if title != "" and title[-1] == " ":
            title = title[:-1]
        
        title += dataframe["Year"][i]
        
        cleanname.append(singer)
        cleantitle.append(title)        
    
    dataframe["Performer"] = cleanname
    dataframe["Song"] = cleantitle
    
    return dataframe

        
def crawler(artist, title, trial):
    '''
    Web crawler that accesses a lyric information website, extracts lyrics,
    and processes them into analysis-friendly format
    
    Inputs:
        artist (str): artist name
        title (str): song title
        trial (int): indicator for number of crawling tried thus far
        (website to access depends on this)
    
    Returns:
        tag (str): lyrics data extracted from html document
    '''
    copy = (artist, title)
    artist = artist.lower().replace(" ", guide[trial][2])
    title = title.replace(" ", guide[trial][2])    
    pm = urllib3.PoolManager(cert_reqs = "CERT_REQUIRED", 
                             ca_certs = certifi.where())
    
    if trial == 0:
        site = ("http://www.songlyrics.com/" + artist + "/" + title + 
                "-lyrics/")
    
    elif trial == 1:
        site = ("https://www.lyricsondemand.com/" + artist[0] + "/" + 
                artist + "lyrics/" + title + "lyrics.html")
    
    elif trial == 2:
        site = ("https://lyrics.az/" + artist + "/-/" + title + ".html")
    
    else:
        artist = artist[0].upper() + artist[1:]
        site = ("https://genius.com/" + artist + "-" + title + "-lyrics")
        
    soup = bs4.BeautifulSoup(pm.urlopen(url = site, method = "GET").data,
                             features = "lxml")
    
    taglist = soup.find_all(guide[trial][0], class_ = guide[trial][1])

    if len(taglist) == 0:
        return None

    tag = lyric_cleaner(unidecode.unidecode(taglist[0].text), copy)
        
    if len(tag) < 200:
        return None
    
    for item in filters:
        if item in tag:
            return None
    
    return tag
    

def lyric_cleaner(tag, copy):
    '''
    Processes the raw lyrics from the website into analysis-friendly format
    by removing irrelevant and irregular parts
    (Involves some hard-coding since many of irregularity patterns are detected
    by manual insepction)
    
    Input:
        tag (str): raw lyrics text
        copy (tuple of str): stores non-edited strings of artist name and title
        
    Returns:
        processed lyrics (str)
    '''
    tag = tag.replace("<br />", " ")
    tag = tag.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    tag = tag.replace(copy[0] + " Miscellaneous " + copy[1], "")
    tag = tag.replace(copy[0] + " Miscellaneous ", "")
    tag = tag.replace("iscellaneous", "")
    tag = tag.replace("CHORUS", " ")
    tag = tag.replace("REPEAT", " ")
    tag = tag.replace(":", " ")
    tag = tag.replace("TWICE", " ")
       
    tag = re.sub(r"\[[^[]*\]", "", tag)
    tag = re.sub(r"\([^(]*\)", "", tag)
    tag = re.sub(r"\{[^{]*\}", "", tag)
    tag = re.sub(r"\<[^<]*\>", "", tag)
    tag = re.sub("try\{.*", "", tag)
    tag = re.sub("[0-9\/\|\#\-]+", "", tag)
    tag = re.sub("[\s][\s]+", " ", tag)
    
    if tag[:len(copy[0])] == copy[0]:
        tag = tag[len(copy[0]):]
    
    tag = re.sub(".*\]\s","",tag)
    
    return tag
    

def freq_count(n, df):
    '''
    Counts Billboard chart-in frequency of each artist and selects only artists
    of frequency higher than the given threshold
    
    Inputs:
        n (int): frequency threshold to be selected (n or more times chart-in)
        dataframe (pandas df): Billboard data
    
    Returns:
        result (list): list of names of artists who achieved chart-in for
        more than n times
    '''
    count_raw, result = {}, []
    
    for i in range(len(df)):

        performer, title = df["Performer"][i], df["Song"][i].lower()
        
        if title != "":
            
            if not performer in count_raw:
                count_raw[performer] = {title[:-4]:title[-4:]}
            
            else:
                count_raw[performer][title[:-4]] = title[-4:]
        
    for name in count_raw:
        if len(count_raw[name]) >= n:
            result.append(name)
    
    return count_raw, result


### main workspace
billboard = pd.read_csv("bill_board.csv")
billboard["Year"] = billboard["WeekID"].apply(lambda x: x[-4:])
billboard = billboard.sort_values(by = ["Year"])
billboard = clean(billboard)

#Can choose any number here (instead of 45)
count, top_artists = freq_count(45, billboard)
musician_list = []

for name in top_artists:

    musician = artist(name)
    songlist = count[name].keys()
    
    for item in songlist:
        
        for trial in range(4):
            
            lyric = crawler(name, item, trial)
            
            if lyric != None:
                music = song(name, item, lyric, count[name][item])
                musician.songs.append(music)
                break

    musician_list.append(musician)
    

with open("artist_lyrics.txt", "w") as f:
    
    f.write("artist|title|year|lyric\n")
    
    for musician in musician_list:
        for song in musician.songs:
            f.write("{:s}|{:s}|{:s}|{:s}\n".format(
                    musician.name, song.title, song.year, song.lyric))
            
    f.close()