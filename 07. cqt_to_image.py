
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from io import BytesIO
import cv2
import os 


def cqt_to_image(df):
    
    specs = []
    
    for i in tqdm(range(df.shape[0]), desc="Convertendo CQT para Espectrograma"):
        
        # Cria uma figura sem eixos ou bordas para salvar apenas o espectrograma
        fig = plt.figure(figsize=(1.28, 1.28), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Converte a amplitude para decibéis e mostra o espectrograma
        db = librosa.amplitude_to_db(np.abs(df['cqt'][i]), ref=1.0, top_db=80)
        librosa.display.specshow(db,
                                 sr=df['sr'][i],
                                 x_axis='time',
                                 y_axis='cqt_note',
                                 cmap='gray', 
                                 vmin=-80, vmax=0)

        
        # Salva a figura em um buffer de memória em vez de um arquivo
        buffer = BytesIO()
        plt.savefig(buffer,
                    format='png',
                    dpi=100,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)
        
        # Lê a imagem do buffer com o OpenCV, redimensiona e normaliza
        buffer.seek(0)
        img = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1) # Adiciona dimensão de canal

        
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

    # -------- SALVANDO ESPECTROGRAMAS ALEATORIOS PARA USAR COMO EXEMPLOS --------------

    print("\n--- Salvando 5 espectrogramas aleatórios como exemplo ---")

    n_amostras = 5
    pasta_saida = "espectrogramas_exemplo"
    os.makedirs(pasta_saida, exist_ok=True) 
    amostras_aleatorias = df.sample(n=n_amostras, random_state=42)

    for index, row in amostras_aleatorias.iterrows():
        id_musica = row['id']
        imagem = row['cqt_image']
        rotulos = row['instrument']

        imagem_para_salvar = (imagem * 255).astype(np.uint8)
        
        nome_arquivo = f"{id_musica}_espec.png"
        caminho_completo = os.path.join(pasta_saida, nome_arquivo)

        cv2.imwrite(caminho_completo, imagem_para_salvar)
        print(f"ID: {id_musica} | Rótulos: {rotulos} -> Imagem salva em '{caminho_completo}'")
