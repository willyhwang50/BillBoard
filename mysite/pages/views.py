import json
import traceback
import sys
import csv
import os
import main_task

from django.http import HttpResponse
from django.shortcuts import render
from django import forms

RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'res')

class Lyric(forms.Form):
    '''
    Constructor for lyric_view page
    '''
    query = forms.CharField(max_length= 5000, label='Your lyrics here', required= True)
    show_args = forms.BooleanField(label='Show args_to_ui',
                                   required=False)

def lyric_view(request):
    '''
    Creates a webpage that inputs lyrics from the user and displays the most 
    similar artist and the decade
    '''
    context = {}
    res = None
    if request.method == 'GET':
        form = Lyric(request.GET)

        if form.is_valid():
            args = {}
            if form.cleaned_data['query']:
                args['lyrics'] = form.cleaned_data['query']

            if form.cleaned_data['show_args']:
                context['args'] = 'args_to_ui = ' + json.dumps(args, indent=2)

            try:
                res = main_task.compute_similarity(args['lyrics'])
            except Exception as e:
                print('Exception caught')
                bt = traceback.format_exception(*sys.exc_info()[:3])
                context['err'] = """
                An exception was thrown in main_task:
                <pre>{}
{}</pre>
                """.format(e, '\n'.join(bt))

                res = None
    else:
        form = Lyric()

    # Handle different responses of res
    if res is None:
        context['result'] = None
    elif isinstance(res, str):
        context['result'] = None
        context['err'] = res
        result = None

    else:
        artist, decade = res

        context['artist'] = artist
        context['decade'] = decade

    context['form'] = form

    return render(request, 'create.html', context)


class Artist(forms.Form):
    '''
    Constructor for artist_view page
    '''
    query = forms.CharField(max_length= 100, label='Your artist here', required= True)
    show_args = forms.BooleanField(label='Show args_to_ui',
                                   required=False)

def artist_view(request):
    '''
    Creates a webpage that inputs artist from the user and displays the most 
    similar artist to the inputted artist
    '''

    context = {}
    res = None
    if request.method == 'GET':
        form = Artist(request.GET)

        if form.is_valid():
            args = {}
            if form.cleaned_data['query']:
                args['artist'] = form.cleaned_data['query']

            if form.cleaned_data['show_args']:
                context['args'] = 'args_to_ui = ' + json.dumps(args, indent=2)

            try:
                res = main_task.most_similar_artist(args['artist'])
            except Exception as e:
                print('Exception caught')
                bt = traceback.format_exception(*sys.exc_info()[:3])
                context['err'] = """
                An exception was thrown in main_task:
                <pre>{}
{}</pre>
                """.format(e, '\n'.join(bt))

                res = None
    else:
        form = Artist()

    # Handle different responses of res
    if res is None:
        context['result'] = None
    
    elif res not in list(main_task.data['artist'].unique()):
        context['result'] = None
        context['err'] = res
        result = None
    
    else:
        artist = res

        context['artist'] = artist

    context['form'] = form

    return render(request, "artist.html", context)

class Posneg(forms.Form):
    '''
    Constructor for pos_neg page
    '''
    query = forms.CharField(max_length= 100, label='Your artist/year here', required= True)
    show_args = forms.BooleanField(label='Show args_to_ui',
                                   required=False)


def posneg_view(request):
    '''
    Creates a webpage that inputs artist/year from the user and displays songs 
    with the most positive/negative sentiment of the given condition
    '''

    context = {}
    res = None
    if request.method == 'GET':
        form = Posneg(request.GET)

        if form.is_valid():
            args = {}
            if form.cleaned_data['query']:
                args['df'] = form.cleaned_data['query']

            if form.cleaned_data['show_args']:
                context['args'] = 'args_to_ui = ' + json.dumps(args, indent=2)

            try:
                res = main_task.most_positive_negative(args['df'])
            except Exception as e:
                print('Exception caught')
                bt = traceback.format_exception(*sys.exc_info()[:3])
                context['err'] = """
                An exception was thrown in main_task:
                <pre>{}
{}</pre>
                """.format(e, '\n'.join(bt))

                res = None
    else:
        form = Posneg()

    # Handle different responses of res
    if res is None:
        context['result'] = None
    
    elif isinstance(res, str):
        context['result'] = None
        context['err'] = res
        result = None
    
    else:
        context['positive_artist'] = res['artist'][0]
        context['positive_title'] = res['title'][0]
        context['positive_year'] = res['year'][0]

        context['negative_artist'] = res['artist'][1]
        context['negative_title'] = res['title'][1]
        context['negative_year'] = res['year'][1]

    context['form'] = form

    return render(request, "posneg.html", context)    

