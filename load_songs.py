import os
import librosa
import pandas as pd
from tqdm import tqdm

def load_songs(path, limit_songs = False, max_songs = 50):
    songs = []
    
    # Primeiro, coletamos todos os arquivos .ogg
    songs_folder = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.ogg'):
                songs_folder.append((root, file))
    
    if limit_songs:
        songs_folder = songs_folder[:max_songs]

    for root, file in tqdm(songs_folder, desc="Carregando músicas"):
        file_path = os.path.join(root, file)
        
        try:
            song, sr = librosa.load(file_path, sr=None)
            
            song_id = os.path.splitext(file)[0]
            
            songs.append({
                'id': song_id,
                'audio': song,
                'sample_rate': sr
            })
            
        except Exception as e:
            print(f"Erro ao processar o arquivo {file_path}: {e}")
    
    df = pd.DataFrame(songs)
    return df
                    

if __name__ == "__main__":
    path = "D:/Documentos/UTFPR/TCC/openmic-2018/audio"
    
    # Carrega apenas 50 músicas
    df = load_songs(path, limit_songs = False, max_songs = 5000)
    
    print(df.head())
    print(f"Total de músicas carregadas: {len(df)}")
    
    df.to_pickle(path="./data/songs_loaded.pkl")