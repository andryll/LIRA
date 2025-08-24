
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Gera o índice de jaccard dos resultados

if __name__ == "__main__":

    instrument_list = ['clarinet', 'flute', 'trumpet', 'saxophone', 'voice',
                   'accordion', 'ukulele', 'mallet_percussion', 'piano',
                   'guitar', 'mandolin', 'banjo', 'synthesizer', 'trombone',
                   'organ', 'drums', 'bass', 'cymbals', 'cello', 'violin']
    
    df_path = "./data/probas_results_ECC_XGB.pkl"
    
    with open("./data/probas_results_ECC_XGB.pkl", "rb") as f:
        df = pickle.load(f)
    
    df = pd.concat(df.values())


    df['probas_instrument_ecc'] = df['probas_instrument_ecc'].apply(
        lambda lista: [1 if item > 0.5 else 0 for item in lista]
    )

    df = df.applymap(
        lambda binary_list: [instrument_list[i] for i, is_present in enumerate(binary_list) if is_present == 1]
    )

    
    df['n_instruments'] = df['probas_instrument_ecc'].apply(len)
    count = df['n_instruments'].value_counts().sort_index()
    
    plt.figure()
    bars = plt.bar(count.index, count.values, color='crimson')
    plt.bar_label(bars, labels=count.values, padding=3)
    plt.title('Quantidade de Músicas por Número de Instrumentos', fontweight = 'bold')
    plt.xlabel('Número de Instrumentos (n)')
    plt.ylabel('Quantidade de Músicas')
    plt.xticks(rotation=0)
    plt.ylim([0, 9200])
    # plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Mostrar o gráfico
    plt.tight_layout()
    plt.savefig("./graphs/musica_x_nrotulos_filt.pdf")
    
    print(f"Cardinalidade de Rótulo: {np.mean(df['n_instruments'])}")
    
    mlb = MultiLabelBinarizer()
    binary_data = mlb.fit_transform(df['probas_instrument_ecc'])
    
    jaccard_dist = pairwise_distances(binary_data.T, metric = 'jaccard')
    jaccard_sim = 1 - jaccard_dist
    jaccard_df = pd.DataFrame(jaccard_sim, index = mlb.classes_, columns=mlb.classes_)
    
    import seaborn as sns
    
    mask = np.triu(np.ones_like(jaccard_df, dtype=bool))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        jaccard_df,
        mask=mask, 
        annot=True,
        annot_kws={'fontsize': 7},
        cmap='mako',
        fmt='.2f',
        vmin=0,
        vmax=0.6,
        linewidths=0.6
    )
    plt.title('Índice de Jaccard entre os instrumentos para o ECC + XGB', fontweight='bold')
    plt.tight_layout()
    plt.savefig("./graphs/jaccard_filt_ecc.pdf")
         