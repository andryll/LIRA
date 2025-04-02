# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:57:11 2024

@author: André
"""

import pandas as pd
import matplotlib.pyplot as plt


path = "./../openmic-2018/openmic-2018-aggregated-labels.csv"


data = pd.read_csv(path)

data.head()

instruments = data['instrument'].unique()

print(instruments)

woodwinds = ['clarinet', 'flute', 'saxophone']
brass = ['trumpet', 'trombone']
plucked_strings = ['ukulele', 'guitar', 'banjo', 'bass', 'mandolin']
bowed_strings = ['cello', 'violin']
percussive_string = ['piano']
percussion = ['drums', 'cymbals', 'mallet_percussion']
aerophones = ['accordion', 'organ']
electronic = ['synthesizer']
vocals = ['voice']

count = data['instrument'].value_counts()

print(count)

df = data.groupby('sample_key')['instrument'].apply(list).reset_index()

rate = count/20000

print(rate)

rate.plot(kind='bar')
plt.title('Taxa de presença dos instrumentos')
plt.xlabel("Instrumento")
plt.ylabel('Taxa')

plt.show()

df.to_pickle(path = "./data/openmic-2018-labels.pkl")