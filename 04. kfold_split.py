
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import json

# Realizando um 10Fold Split após a filtragem
if __name__ == "__main__":
    
    df_path = "D:/Documentos/UTFPR/TCC/LIRA/data/openmic-2018-labels-filtrado.pkl"    
    df = pd.read_pickle(df_path)
    
    df['instrument_str'] = df['instrument'].apply(lambda x: '_'.join(sorted(x)))
    # df_train, df_test = train_test_split(df, test_size=0.1, random_state=seed, stratify=df["instrument_str"])

    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=42)
    
    # Salvar os ids de cada fold
    folds = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df["instrument_str"])):
        train_ids = df.iloc[train_idx]["sample_key"].tolist()
        test_ids = df.iloc[test_idx]["sample_key"].tolist()
        folds[fold_idx] = {"train_ids": train_ids, "test_ids": test_ids}
    
    # Exemplo de visualização
    for k, v in folds.items():
        print(f"Fold {k}:")
        print("  Train IDs:", v["train_ids"])
        print("  Test IDs:", v["test_ids"])
        
    with open("./data/kfold_split.json", "w", encoding='utf-8') as f:
        json.dump(folds, f, ensure_ascii=False, indent=4)