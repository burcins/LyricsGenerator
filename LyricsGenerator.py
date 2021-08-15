import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 
import re, warnings, flask, joblib
from tensorflow import keras
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from unidecode import unidecode

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


#tickers = ['sezen-aksu']

def select_singer():
    ticker = input('Please write singer name in this lowercase characters with adding "-" between name and surname (exp: sezen-aksu)\n','\033[1m', 'Note that this script gathered data from "https://sarki.alternatifim.com/" and it may have some copy rights.','\033[0m')
       
    url = 'https://sarki.alternatifim.com/sarkici/'


    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
         'Referer': 'https://cssspritegenerator.com',
         'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
         'Accept-Encoding': 'none',
         'Accept-Language': 'en-US,en;q=0.8',
         'Connection': 'keep-alive'}
    
    


    w_url = url + ticker

    response = urlopen(Request(url = w_url, headers=hdr))
    lyric_table = BeautifulSoup(response, 'html')
    song_names = []
    
    for row in lyric_table.findAll('li'):
        if row.a is None:  
            print('Input is wrong!')
            return
        else:
            song_names.append(unidecode(row.a.text))

    if len(song_names) < 5:
        print('Not enough song to analyse')
        return

    for idx,name in enumerate(song_names):
        song_names[idx] = re.sub(r'\W+',  "-", name.lower())
        if song_names[idx].endswith('-'):
            song_names[idx] = song_names[idx][:-1]  
    if len(ticker.split('-')) >1:
        print('Analysed', str(ticker).split('-')[0].upper(), str(ticker).split('-')[1].upper(), 'songs : ')  
    else: 
        print('Analysed', str(ticker).split('-')[0].upper(), 'songs : ')  
        
    for song in song_names:
        print(song)

    all_lyrics = []


    for songname in song_names:
        w_url = url + ticker + '/' + songname
        response = urlopen(Request(url = w_url, headers=hdr))
        lyric = BeautifulSoup(response, 'html')
        lyr = []

        for div in lyric.findAll('br'):
            lyr.append(div.next_sibling)

        lyr = [l for l in lyr if not str(l).startswith('\n') and not str(l).startswith('<div') and l != None and 'href=' not in str(l)]
        lyr = [str(l).split('\n')[0] for l in lyr]
        if len(lyr) > 2:
            lyr = [l.strip().lower() for l in lyr]
            for l in lyr:
                all_lyrics.append(l)
    print('\n')
    print("First 20 lines of lyrics :")
    for i in range(0,20):
        print(all_lyrics[i])   

    print('\n')
    print('Total length of lyrics', len(all_lyrics))
    # Tokenizing lyrics 
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(all_lyrics)
    total_words = len(tokenizer.word_index) + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in all_lyrics:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)


    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

    label = ku.to_categorical(label, num_classes=total_words)
    
    print('lyrics gathered and cleaned, now its time for model development!')
    #model development
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100,activation="tanh"))
    model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    #model training
    history = model.fit(predictors, label, epochs=100, verbose=0)    
    
    print('Model is Ready to predict!')
    
    seed_text = input('Write a few words to let it extend lyrics : ')
    next_words = 100

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        if len((seed_text).split(' ')) % 5 == 0:
            seed_text += "\n"


    print(seed_text)