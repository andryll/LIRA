# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 09:26:56 2025

@author: andry
"""

import pandas as pd
from ECC import ECC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score, classification_report, hamming_loss, accuracy_score, precision_recall_curve, average_precision_score
import time
from xgboost import XGBClassifier
from collections import OrderedDict
import pickle
from datetime import timedelta



#########################################################################################################################

def ecc_classifier(df, folds, algo="XGB", seed=42):
    
    print("Iniciando Classificação\n")
    start = time.time()
    
    # Binarizar os rótulos
    # original_label_order_f = list(OrderedDict.fromkeys(label for sample in df["families"] for label in sample))
    original_label_order_i = list(OrderedDict.fromkeys(label for sample in df["instrument"] for label in sample))
    
    # mlb_f = MultiLabelBinarizer(classes=original_label_order_f)
    mlb_i = MultiLabelBinarizer(classes=original_label_order_i)
    
    resultados_seeds = {}
    
    for fold_num in range(len(folds)):
        print(f"Rodando a seed {fold_num}")
        train_idx = folds[fold_num]['train_ids']
        test_idx = folds[fold_num]['test_ids']
        
        df_train = df[df['id'].isin(train_idx)].copy()
        df_test = df[df['id'].isin(test_idx)].copy()
        
        # y_train_f = mlb_f.fit_transform(df_train['families'])
        # y_test_f = mlb_f.transform(df_test['families'])
        
        y_train_i = mlb_i.fit_transform(df_train['instrument'])
        y_test_i = mlb_i.transform(df_test['instrument'])
        
    
        if algo == "RF":
            base_classifier = RandomForestClassifier(random_state=fold_num)
        elif algo == "XGB":
            base_classifier = XGBClassifier(eval_metric='logloss', random_state=fold_num)
        # elif algo == "AUTOG":
        #     base_classifier = AutoGluonWrapper(problem_type='multiclass')
        else:
            raise ValueError("Algoritmo não suportado. Use 'RF', 'XGB' ou 'AUTOG'.")
        
     
        df_resultados = pd.DataFrame()
        df_resultados['id'] = df_test['id'].values
        
        # Classifying Families
        print("Classificando famílias...")
        # ecc = ECC(model=base_classifier, n_chains=10, n_jobs=1)
        # ecc.fit(df_train.drop(columns=['id', 'instrument', 'families']), y_train_f)
        # probas_f = ecc.predict_proba(df_test.drop(columns=['id', 'instrument', 'families']))
        # df_resultados['probas_family_ecc'] = list(probas_f)
        
        # Classifying Instruments
        print("Classificando instrumentos...")
        ecc = ECC(model=base_classifier, n_chains=10, n_jobs=5)
        ecc.fit(df_train.drop(columns=['id', 'instrument', 'families']), y_train_i)
        probas_i = ecc.predict_proba(df_test.drop(columns=['id', 'instrument', 'families']))
        df_resultados['probas_instrument_ecc'] = list(probas_i)
        
        
        # Adicionando os rótulos verdadeirosDTE + Médias e SDs
        # df_resultados['family_label'] = list(y_test_f)
        df_resultados['instrument_label'] = list(y_test_i)
        resultados_seeds[fold_num] = df_resultados
    
    end = time.time()
    tempo_formatado = str(timedelta(seconds=int(end- start)))
    print(f"Classificação Finalizada! Tempo: {tempo_formatado}")
    
    return resultados_seeds

############################################################################################################


    
#########################################################################################

if __name__ == "__main__":

        df_path = "./data/features.pkl"
        folds_path = "./data/kfold_split.pkl"
        
        # Manipulando o dataset para separar conjuntos de treino e teste
        df = pd.read_pickle(df_path)
        with open(folds_path, "rb") as f:
            folds = pickle.load(f)
              
        results = ecc_classifier(df, folds, algo = "RF", seed = 42)
        
        with open("./data/probas_results_ECC_RF.pkl", "wb") as f:
            pickle.dump(results, f)
            
        print("Finalizando programa")