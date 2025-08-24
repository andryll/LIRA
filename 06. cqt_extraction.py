import librosa
import pandas as pd
from tqdm import tqdm

# Extraindo o CQT com o librosa
def extract_cqt(songs_path, labels_path):
    
    cqt_df = []
    
    songs_df = pd.read_pickle(songs_path)
    labels_df = pd.read_pickle(labels_path)
    songs_df = songs_df[songs_df['id'].isin(labels_df['sample_key'])].reset_index(drop=True)
    
    hop_lenght = 512
    
    for i in tqdm(range(songs_df.shape[0]), desc = "Extraindo CQT Harmônico"):
        
        cqt = librosa.feature.chroma_cqt(y = songs_df['audio'][i],
                                         sr = songs_df['sample_rate'][i], 
                                         hop_length = hop_lenght,
                                         n_octaves = 7,
                                         bins_per_octave = 36)
        
        cqt_df.append({
            'id': songs_df['id'][i],
            'cqt': cqt,
            'sr': songs_df['sample_rate'][i]
        })
        
    df = pd.DataFrame(cqt_df)
    df = pd.merge(df, labels_df, left_on = 'id', right_on = 'sample_key', how = 'left')
    df = df.drop(columns=['sample_key'])

    print("\n\n")

    return df


if __name__ == "__main__":
   
    labels_path = "./data/openmic-2018-labels-filtrado.pkl"
    songs_path = "./data/songs_loaded.pkl"
    
    df = extract_cqt(songs_path, labels_path)

    print("Extração dos CQTs Completa!")
    
    df.to_pickle(path="./data/cqt.pkl")

    print("Pickle Salvo!")