import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer

# Calculando Cardinalidade dos Rótulos e Diagrama de Jaccard

if __name__ == "__main__":
    
    labels_path = "./data/openmic-2018-labels-filtrado.pkl"
    
    df = pd.read_pickle(labels_path)
    
    df['n_instruments'] = df['instrument'].apply(len)
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
    binary_data = mlb.fit_transform(df['instrument'])
    
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
    plt.title('Índice de Jaccard entre os instrumentos', fontweight='bold')
    plt.tight_layout()
    plt.savefig("./graphs/jaccard_filt.pdf")
         
    # Fazendo a mesma coisa pras famílias
    
    df['n_families'] = df['families'].apply(len)
    count = df['n_families'].value_counts().sort_index()
    
    plt.figure()
    bars = plt.bar(count.index, count.values, color='skyblue')
    plt.bar_label(bars, labels=count.values, padding=3)
    plt.title('Quantidade de Músicas por Número de Famílias')
    plt.xlabel('Número de Famílias (n)')
    plt.ylabel('Quantidade de Músicas')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Mostrar o gráfico
    plt.tight_layout()
    plt.show()
    
    print(f"Cardinalidade de Rótulo: {np.mean(df['n_families'])}")
    
    mlb = MultiLabelBinarizer()
    binary_data = mlb.fit_transform(df['families'])
    
    jaccard_dist = pairwise_distances(binary_data.T, metric = 'jaccard')
    jaccard_sim = 1 - jaccard_dist
    jaccard_df = pd.DataFrame(jaccard_sim, index = mlb.classes_, columns=mlb.classes_)
    
    mask = np.triu(np.ones_like(jaccard_df, dtype=bool))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        jaccard_df,
        mask=mask,
        annot=True,
        annot_kws={'fontsize': 7},
        cmap='mako',
        fmt='.2f',
        # vmin=0,
        # vmax=0.6,
        linewidths=0.6
    )
    plt.title('Índice de Jaccard entre as famílias')
    plt.show()
    