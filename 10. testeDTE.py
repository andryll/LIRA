
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report, hamming_loss, accuracy_score, precision_recall_curve, average_precision_score
import time
from collections import OrderedDict
import pickle
from datetime import timedelta
from DTE import DTE
import json


def dte_classifier(df, folds, seed=42):
    
    features = True
    output_space_features = True
    tree_embedding_features = False
    
    print("Iniciando Classificação\n")
    start = time.time()
    
    # Binarizar os rótulos
    # original_label_order_f = list(OrderedDict.fromkeys(label for sample in df["families"] for label in sample))
    original_label_order_i = list(OrderedDict.fromkeys(label for sample in df["instrument"] for label in sample))
    
    # mlb_f = MultiLabelBinarizer(classes=original_label_order_f)
    mlb_i = MultiLabelBinarizer(classes=original_label_order_i)
    
    resultados_seeds = {}
    
    for fold_num in range(len(folds)):
        print(f"\nRodando a seed {fold_num}")
        train_idx = folds[fold_num]['train_ids']
        test_idx = folds[fold_num]['test_ids']
        
        df_train = df[df['id'].isin(train_idx)].copy()
        df_test = df[df['id'].isin(test_idx)].copy()
        
        # y_train_f = mlb_f.fit_transform(df_train['families'])
        # y_test_f = mlb_f.transform(df_test['families'])
        
        y_train_i = mlb_i.fit_transform(df_train['instrument'])
        y_test_i = mlb_i.transform(df_test['instrument'])        
        df_resultados = pd.DataFrame()
        # df_resultados['id'] = df_test['id'].values
        
        print((df_train.drop(columns=['id', 'instrument', 'families']).head()))
        print(y_train_i.shape)


        # Classificando Instrumentos
        print("Classificando instrumentos...")
        dte = DTE(task="mlc", features=features, output_space_features=output_space_features, tree_embedding_features=tree_embedding_features)
        dte.fit(df_train.drop(columns=['id', 'instrument', 'families']), pd.DataFrame(y_train_i))
        probas_i = dte.predict(df_test.drop(columns=['id', 'instrument', 'families']))
        
        print(f"Probas: {probas_i}")
        
        print("============================================")
        print(f"Tipo do probas_i: {type(probas_i)}")
        print(f"Shape do probas_i: {probas_i.shape}")
        print("============================================")
        print(f"Tipo do y_test_i: {type(y_test_i)}")
        print(f"Shape do y_test_i: {y_test_i.shape}")
        
        
        df_resultados = pd.DataFrame({
            'probas_instrument_dte': probas_i.values.tolist(),
            'instrument_label': [list(x) for x in y_test_i]
        })

        
        resultados_seeds[fold_num] = df_resultados
    
    end = time.time()
    tempo_formatado = str(timedelta(seconds=int(end - start)))
    print(f"\nClassificação Finalizada! Tempo: {tempo_formatado}")
    
    return resultados_seeds

#########################################################################################

if __name__ == "__main__":

        df_path = "./data/features.csv"
        folds_path = "./data/kfold_split.pkl"
        
        # Manipulando o dataset para separar conjuntos de treino e teste
        df = pd.read_csv(df_path)
        
        df['instrument'] = df['instrument'].apply(json.loads)
        df['families'] = df['families'].apply(json.loads)

        
        with open(folds_path, "rb") as f:
            folds = pickle.load(f)
              
        results = dte_classifier(df, folds, seed = 42)
        
        with open("./data/results_dte_xte.pkl", "wb") as f:
            pickle.dump(results, f)
            
        print("Finalizando programa")