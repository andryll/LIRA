# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 08:58:21 2025

@author: andry
"""

import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def extract_features(songs_path, labels_path):
    
    features_df = []
    
    songs_df = pd.read_pickle(songs_path)
    labels_df = pd.read_pickle(labels_path)
    
    frame_lenght = 1024
    hop_lenght = int(frame_lenght/2)
    
    for i in tqdm(range(songs_df.shape[0]), desc = "Extraindo features"):
        
        
        centroid = librosa.feature.spectral_centroid(y = songs_df['audio'][i], sr = songs_df['sample_rate'][i],
                                                     n_fft = frame_lenght, hop_length = hop_lenght, window = 'hann')
        rolloff05 = librosa.feature.spectral_rolloff(y = songs_df['audio'][i], sr = songs_df['sample_rate'][i], roll_percent = 0.05,
                                                     n_fft = frame_lenght, hop_length = hop_lenght, window = 'hann')
        rolloff95 = librosa.feature.spectral_rolloff(y = songs_df['audio'][i], sr = songs_df['sample_rate'][i], roll_percent = 0.95,
                                                     n_fft = frame_lenght, hop_length = hop_lenght, window = 'hann')
        bandwidth = librosa.feature.spectral_bandwidth(y = songs_df['audio'][i], sr = songs_df['sample_rate'][i],
                                                     n_fft = frame_lenght, hop_length = hop_lenght, window = 'hann')
        flatness = librosa.feature.spectral_flatness(y = songs_df['audio'][i], n_fft = frame_lenght,
                                                     hop_length = hop_lenght, window = 'hann')
        contrast = librosa.feature.spectral_contrast(y = songs_df['audio'][i], sr = songs_df['sample_rate'][i],
                                                     n_fft = frame_lenght, hop_length = hop_lenght, window = 'hann',
                                                     n_bands = 5, fmin = min(200, songs_df['sample_rate'][i] * 0.1))
        zcr = librosa.feature.zero_crossing_rate(y = songs_df['audio'][i], frame_length = frame_lenght, hop_length = hop_lenght)
        
        mfcc = librosa.feature.mfcc(y = songs_df['audio'][i], sr = songs_df['sample_rate'][i], n_mfcc = 13,
                                                     n_fft = frame_lenght, hop_length = hop_lenght, window = 'hann')
        
        features_df.append({
            'id': songs_df['id'][i],
            
            'centroid_mean': np.mean(centroid),
            'centroid_sd': np.std(centroid),
            
            'rolloff05_mean': np.mean(rolloff05),
            'rolloff05_sd': np.std(rolloff05),
            
            'rolloff95_mean': np.mean(rolloff95),
            'rolloff95_sd': np.std(rolloff95),
            
            'bandwidth_mean': np.mean(bandwidth),
            'bandwidth_sd': np.std(bandwidth),
            
            'flatness_mean': np.mean(flatness),
            'flatness_sd': np.std(flatness),
            
            'contrast_mean': np.mean(contrast),
            'contrast_sd': np.std(contrast),
            
            'zcr_mean': np.mean(zcr),
            'zcr_sd': np.std(zcr),
            
            'mfcc01_mean': np.mean(mfcc[0]),
            'mfcc01_sd': np.std(mfcc[0]),
            
            'mfcc02_mean': np.mean(mfcc[1]),
            'mfcc02_sd': np.std(mfcc[1]),
            
            'mfcc03_mean': np.mean(mfcc[2]),
            'mfcc03_sd': np.std(mfcc[2]),
            
            'mfcc04_mean': np.mean(mfcc[3]),
            'mfcc04_sd': np.std(mfcc[3]),
            
            'mfcc05_mean': np.mean(mfcc[4]),
            'mfcc05_sd': np.std(mfcc[4]),
            
            'mfcc06_mean': np.mean(mfcc[5]),
            'mfcc06_sd': np.std(mfcc[5]),
            
            'mfcc07_mean': np.mean(mfcc[6]),
            'mfcc07_sd': np.std(mfcc[6]),
            
            'mfcc08_mean': np.mean(mfcc[7]),
            'mfcc08_sd': np.std(mfcc[7]),
            
            'mfcc09_mean': np.mean(mfcc[8]),
            'mfcc09_sd': np.std(mfcc[8]),
            
            'mfcc10_mean': np.mean(mfcc[9]),
            'mfcc10_sd': np.std(mfcc[9]),
            
            'mfcc11_mean': np.mean(mfcc[10]),
            'mfcc1_sd': np.std(mfcc[10]),
            
            'mfcc12_mean': np.mean(mfcc[11]),
            'mfcc12_sd': np.std(mfcc[11]),
            
            'mfcc13_mean': np.mean(mfcc[12]),
            'mfcc13_sd': np.std(mfcc[12])
        })
    
    df = pd.DataFrame(features_df)
    df = pd.merge(df, labels_df, left_on = 'id', right_on = 'sample_key', how = 'left')
    df = df.drop(columns=['sample_key'])
    
    non_normalized = ['id', 'instrument']
    
    scaler = MinMaxScaler()
    df[df.columns[~df.columns.isin(non_normalized)]] = scaler.fit_transform(df[df.columns[~df.columns.isin(non_normalized)]])

    print("\n\n")

    return df


if __name__ == "__main__":
   
    labels_path = "./data/openmic-2018-labels.pkl"
    songs_path = "./data/songs_loaded.pkl"
    
    df = extract_features(songs_path, labels_path)

    print("Extração de Features Completa!")
    
    df.to_pickle(path="./data/features.pkl")

    