import pandas as pd
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # Caminhos
    path1 = "./data/new_features/centroid.pkl"
    path2 = "./data/new_features/outras_features1.pkl"
    path3 = "./data/new_features/outras_features2.pkl"
    path4 = "./data/new_features/mfcc.pkl"

    # Carregar os DataFrames
    df1 = pd.read_pickle(path1)
    df2 = pd.read_pickle(path2)
    df3 = pd.read_pickle(path3)
    df4 = pd.read_pickle(path4)

    # Função para converter listas em strings temporariamente
    def list_to_str(x):
        if isinstance(x, list):
            return ','.join(map(str, x))
        return str(x)

    # Função para reverter strings para listas
    def str_to_list(x):
        if isinstance(x, str):
            return x.split(',') if x else []
        return x

    # Converter instrument e families para string temporariamente
    for df in [df1, df2, df3, df4]:
        df['instrument'] = df['instrument'].apply(list_to_str)
        df['families'] = df['families'].apply(list_to_str)

    # Juntar os DataFrames
    df_merged = df1.merge(df2, on=['id', 'instrument', 'families'], how='outer')
    df_merged = df_merged.merge(df3, on=['id', 'instrument', 'families'], how='outer')
    df_merged = df_merged.merge(df4, on=['id', 'instrument', 'families'], how='outer')

    # Preencher NaNs com 0
    df_merged = df_merged.fillna(0)

    # Separar colunas categóricas
    categorical_cols = ['id', 'instrument', 'families']
    X = df_merged.drop(columns=categorical_cols)

    # PCA para manter 95% da variância
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X)

    # DataFrame com os componentes principais
    df_pca = pd.DataFrame(X_pca, index=df_merged.index, columns=[f'pca_{i+1}' for i in range(X_pca.shape[1])])

    # Reverter instrument e families para listas
    df_merged['instrument'] = df_merged['instrument'].apply(str_to_list)
    df_merged['families'] = df_merged['families'].apply(str_to_list)

    # Concatenar final
    final_df = pd.concat([df_merged[categorical_cols], df_pca], axis=1)

    # (Opcional) Salvar resultado
    final_df.to_pickle("./data/final_features_pca95.pkl")

    print("Final shape:", final_df.shape)
    print("Columns:", final_df.columns.tolist())
