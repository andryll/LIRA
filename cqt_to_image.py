# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 08:58:21 2025

@author: andry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from io import BytesIO
import cv2


def cqt_to_image(df):
    
    specs = []
    
    for i in tqdm(range(df.shape[0]), desc="Convertendo CQT para Espectrograma"):
        
        fig = plt.figure(figsize=(1.28, 1.28), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(df['cqt'][i]), ref=np.max),
                                 sr = df['sr'][i],
                                 x_axis='time',
                                 y_axis='cqt_note',
                                 cmap='viridis')
        
        buffer = BytesIO()
        plt.savefig(buffer,
                    format='png',
                    dpi=100,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)
        
        buffer.seek(0)
        img = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        
        specs.append({
            'id': df['id'][i],
            'cqt_image': img,
            'instrument': df['instrument'][i]
        })
    
    df_images = pd.DataFrame(specs)

    print("\n\n")

    return df_images



if __name__ == "__main__":
   
    cqt_path = "./data/cqt.pkl"
    
    df_cqt = pd.read_pickle(cqt_path)
    df = cqt_to_image(df_cqt)

    print("CQTs convertidos para Espectrogramas!")
    
    df.to_pickle(path="./data/cqt_image.pkl")
    
    print("Pickle Salvo")

    