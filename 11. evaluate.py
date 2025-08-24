# -*- coding: utf-8 -*-
"""
Script para Avaliação de Modelos de Classificação Multilabel

Este script carrega os resultados de predição de múltiplos algoritmos,
calcula diversas métricas de avaliação, encontra o melhor limiar de
binarização para cada um, realiza testes de significância estatística
(Wilcoxon) e plota gráficos comparativos.

Criado em: 13 de Maio de 2025
@author: andry
Refatorado por: Gemini
"""

import pickle
from collections import defaultdict
from itertools import cycle
from typing import List, Dict, Any, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from sklearn.metrics import (f1_score, classification_report, hamming_loss,
                             accuracy_score, precision_recall_curve,
                             average_precision_score)


# --- Funções de Cálculo de Métricas e Curvas ---

def calcular_curvas_precision_recall(y_test: np.ndarray, y_probs: np.ndarray, n_classes: int) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Calcula as curvas Precision-Recall para cada classe e a média micro.

    Args:
        y_test (np.ndarray): Rótulos verdadeiros (one-hot encoded).
        y_probs (np.ndarray): Probabilidades previstas para cada classe.
        n_classes (int): Número total de classes.

    Returns:
        Tuple[Dict, Dict, Dict, Dict]: Dicionários contendo precisão, recall,
                                       limiares e average precision para cada classe.
    """
    precisao = dict()
    recall = dict()
    limiar = dict()
    avg_precision = dict()

    for i in range(n_classes):
        precisao[i], recall[i], limiar[i] = precision_recall_curve(y_test[:, i], y_probs[:, i])
        avg_precision[i] = average_precision_score(y_test[:, i], y_probs[:, i])

    precisao["micro"], recall["micro"], limiar["micro"] = precision_recall_curve(y_test.ravel(), y_probs.ravel())
    avg_precision["micro"] = average_precision_score(y_test, y_probs, average='micro')

    return precisao, recall, limiar, avg_precision

def plotar_curvas_precision_recall(precisao: Dict, recall: Dict, avg_precision: Dict, n_classes: int, nome_algoritmo: str, lista_instrumentos: List[str]):
    """
    Plota as curvas Precision-Recall para cada classe.

    Args:
        precisao (Dict): Dicionário com os valores de precisão.
        recall (Dict): Dicionário com os valores de recall.
        avg_precision (Dict): Dicionário com os valores de average precision.
        n_classes (int): Número total de classes.
        nome_algoritmo (str): Nome do algoritmo para o título do gráfico.
        lista_instrumentos (List[str]): Lista com os nomes dos instrumentos.
    """
    cores = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue',
                   'teal', 'powderblue', 'darkorchid', 'firebrick',
                   'palegreen', 'gold', 'goldenrod', 'crimson', 'violet',
                   'slategray', 'mediumvioletred', 'darkslateblue',
                   'coral', 'paleturquoise', 'aquamarine', 'lime'])
    lw = 2

    plt.figure(figsize=(12, 8))
    
    if "micro" in recall and "micro" in precisao and "micro" in avg_precision:
        plt.plot(recall["micro"], precisao["micro"], color='gold', lw=lw,
                 label=f'Curva Micro-Média (área = {avg_precision["micro"]:0.2f})')
    else:
        print("Aviso: Não foi possível plotar a curva micro-média devido à ausência de dados.")

    for i, cor in zip(range(n_classes), cores):
        if i in recall and i in precisao and i in avg_precision:
            plt.plot(recall[i], precisao[i], color=cor, lw=lw,
                     label=f'Curva da classe {lista_instrumentos[i]} (área = {avg_precision[i]:0.2f})')
        else:
            print(f"Aviso: Faltam dados para a classe {i}, a curva não será plotada.")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title(f'Curva Precision-Recall para cada classe\n({nome_algoritmo})', fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# --- Funções de Teste Estatístico e Visualização ---

def criar_matriz_significancia(scores_por_fold: Dict[str, List[float]], nomes_algos: List[str], alpha: float = 0.05, zero_method: str = 'pratt') -> pd.DataFrame:
    """
    Cria uma matriz de significância usando o teste de Wilcoxon.

    Args:
        scores_por_fold (Dict[str, List[float]]): Dicionário com os scores de cada algoritmo por fold.
        nomes_algos (List[str]): Lista dos nomes dos algoritmos para comparar.
        alpha (float): Nível de significância.
        zero_method (str): Método para lidar com diferenças zero no teste de Wilcoxon.

    Returns:
        pd.DataFrame: Matriz (DataFrame) com os resultados do teste de significância.
    """
    num_algos = len(nomes_algos)
    matriz_significancia = pd.DataFrame(index=nomes_algos, columns=nomes_algos, dtype=str)

    if not nomes_algos:
        print("\nAVISO: Nenhum algoritmo fornecido para criar a matriz de significância.\n")
        return matriz_significancia

    for i in range(num_algos):
        for j in range(num_algos):
            nome_algo_1 = nomes_algos[i]
            nome_algo_2 = nomes_algos[j]

            if i == j:
                matriz_significancia.loc[nome_algo_1, nome_algo_2] = "-"
                continue

            scores_1 = scores_por_fold.get(nome_algo_1, [])
            scores_2 = scores_por_fold.get(nome_algo_2, [])

            if not scores_1 or not scores_2 or len(scores_1) != len(scores_2):
                matriz_significancia.loc[nome_algo_1, nome_algo_2] = "err_len"
                continue
            
            if len(scores_1) < 2:
                matriz_significancia.loc[nome_algo_1, nome_algo_2] = "ins_data_pair"
                continue

            try:
                _, p_valor = wilcoxon(scores_1, scores_2, zero_method=zero_method, alternative='two-sided')
                
                if p_valor < alpha:
                    mediana_1 = np.median(scores_1)
                    mediana_2 = np.median(scores_2)
                    if mediana_1 > mediana_2:
                        matriz_significancia.loc[nome_algo_1, nome_algo_2] = f"+ (p={p_valor:.3f})"
                    else:
                        matriz_significancia.loc[nome_algo_1, nome_algo_2] = f"- (p={p_valor:.3f})"
                else:
                    matriz_significancia.loc[nome_algo_1, nome_algo_2] = f"~ (p={p_valor:.3f})"
            except ValueError:
                if np.all(np.array(scores_1) == np.array(scores_2)):
                    matriz_significancia.loc[nome_algo_1, nome_algo_2] = "all_diff_zero"
                else:
                    matriz_significancia.loc[nome_algo_1, nome_algo_2] = "err_stat"
    
    return matriz_significancia

def plotar_heatmap_significancia(matriz_sig: pd.DataFrame, titulo: str = "Heatmap Triangular de Significância"):
    """
    Plota um heatmap para visualizar a matriz de significância.

    Args:
        matriz_sig (pd.DataFrame): A matriz de significância gerada.
        titulo (str): Título do gráfico.
    """
    n = len(matriz_sig)
    mascara_significancia = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            val = matriz_sig.iat[i, j]
            if isinstance(val, str) and (val.startswith("+") or val.startswith("-")):
                mascara_significancia[i, j] = True

    mascara_inferior = np.tril(np.ones_like(mascara_significancia, dtype=bool))
    cmap = mcolors.ListedColormap(['lightcoral', 'lightgreen'])
    
    dados_heatmap = np.ma.masked_array(mascara_significancia.astype(int), mask=mascara_inferior)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(dados_heatmap, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(matriz_sig.columns, rotation=45, ha='right')
    ax.set_yticklabels(matriz_sig.index)
    
    ax.grid(False)
    ax.invert_yaxis()
    ax.set_title(titulo, fontweight='bold')
    
    elementos_legenda = [
        Patch(facecolor='lightgreen', edgecolor='k', label='Diferença Significativa (p < α)'),
        Patch(facecolor='lightcoral', edgecolor='k', label='Diferença Não Significativa (p ≥ α)')
    ]
    ax.legend(handles=elementos_legenda, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# --- Funções Auxiliares de Processamento ---

def soma_dicts_com_ajuste(lista_dicts: List[Dict]) -> Dict:
    """Soma os valores de uma lista de dicionários, ajustando para arrays numpy."""
    somas = defaultdict(lambda: None)
    for d in lista_dicts:
        for k, v in d.items():
            if somas[k] is None:
                somas[k] = v.copy() if isinstance(v, np.ndarray) else v
            elif isinstance(v, np.ndarray):
                # Garante que ambos são arrays para a soma
                if isinstance(somas[k], np.ndarray):
                    min_len = min(len(somas[k]), len(v))
                    somas[k] = somas[k][:min_len] + v[:min_len]
            else:
                somas[k] += v
    return somas

def gerar_predicoes(y_probs: np.ndarray, limiar: float, eh_caso_especial: bool = False) -> np.ndarray:
    """
    Gera predições binarizadas a partir das probabilidades.

    Args:
        y_probs (np.ndarray): Array de probabilidades.
        limiar (float): Limiar de binarização.
        eh_caso_especial (bool): Flag para tratar modelos que precisam de pelo
                                 menos uma predição positiva por amostra.

    Returns:
        np.ndarray: Array de predições binarizadas.
    """
    if eh_caso_especial:
        # Para casos como "ECC Binário", o limiar é fixo e há um tratamento especial
        y_pred = (y_probs >= 0.5).astype(int)
        # Se nenhuma classe for prevista, atribui a classe com maior probabilidade
        mask_sem_predicao = np.sum(y_pred, axis=1) == 0
        if np.any(mask_sem_predicao):
            indices_max_prob = np.argmax(y_probs[mask_sem_predicao], axis=1)
            # Cria um array temporário para atualização segura
            temp_y_pred = np.zeros_like(y_pred[mask_sem_predicao])
            temp_y_pred[np.arange(len(indices_max_prob)), indices_max_prob] = 1
            y_pred[mask_sem_predicao] = temp_y_pred
    else:
        y_pred = (y_probs >= limiar).astype(int)
        
    return y_pred

def encontrar_melhor_limiar(dfs: List[pd.DataFrame], coluna_probas: str, eh_caso_especial: bool) -> Tuple[float, float, List[float], List[float], List[Tuple[float, float]]]:
    """
    Encontra o melhor limiar de binarização com base no F1-score macro médio.

    Args:
        dfs (List[pd.DataFrame]): Lista de DataFrames, um para cada fold.
        coluna_probas (str): Nome da coluna com as probabilidades.
        eh_caso_especial (bool): Flag para modelos com lógica de predição especial.

    Returns:
        Tuple: Melhor limiar, melhor F1-score, F1-scores por fold no melhor limiar,
               Subset Accuracy por fold no melhor limiar, e a relação limiar vs F1.
    """
    if eh_caso_especial:
        # Para casos especiais, o limiar é fixo em 0.5 e não há busca
        limiares = [0.5]
    else:
        limiares = np.arange(0.0, 1.0, 0.01)

    melhor_f1 = -1
    melhor_limiar = 0.5
    f1_por_fold_no_melhor_limiar = []
    subset_acc_por_fold_no_melhor_limiar = []
    relacao_limiar_f1 = []

    for limiar in limiares:
        f1_scores_fold_temp = []
        subset_acc_scores_fold_temp = []

        for df_fold in dfs:
            if not isinstance(df_fold, pd.DataFrame) or 'instrument_label' not in df_fold.columns:
                f1_scores_fold_temp.append(np.nan)
                subset_acc_scores_fold_temp.append(np.nan)
                continue
                
            y_test_i = np.array(df_fold['instrument_label'].tolist())
            y_probs_i = np.array(df_fold[coluna_probas].tolist())
            
            y_pred_i = gerar_predicoes(y_probs_i, limiar, eh_caso_especial)
            
            f1_fold = f1_score(y_test_i, y_pred_i, average='macro', zero_division=0)
            f1_scores_fold_temp.append(f1_fold)
            
            subset_acc_fold = accuracy_score(y_test_i, y_pred_i)
            subset_acc_scores_fold_temp.append(subset_acc_fold)
        
        f1_medio = np.nanmean(f1_scores_fold_temp)
        relacao_limiar_f1.append((limiar, f1_medio))

        if f1_medio > melhor_f1:
            melhor_f1 = f1_medio
            melhor_limiar = limiar
            f1_por_fold_no_melhor_limiar = f1_scores_fold_temp.copy()
            subset_acc_por_fold_no_melhor_limiar = subset_acc_scores_fold_temp.copy()
            
    return melhor_limiar, melhor_f1, f1_por_fold_no_melhor_limiar, subset_acc_por_fold_no_melhor_limiar, relacao_limiar_f1

# --- Bloco Principal de Execução ---

if __name__ == "__main__":
    
    # --- Configurações Iniciais ---
    
    lista_instrumentos = [
        'clarinete', 'flauta', 'trompete', 'saxofone', 'vocal', 'sanfona', 
        'ukulelê', 'percussão com baquetas', 'piano', 'violão', 'bandolin', 
        'banjo', 'sintetizador', 'trombone', 'orgão', 'bateria', 'baixo', 
        'pratos', 'violoncelo', 'violino'
    ]

    algoritmos_info = [
        {"nome_amigavel": "DTE + Médias e SDs", "caminho_pkl": "./data/results_dte.pkl", "coluna_probas": "probas_instrument_dte"},
        {"nome_amigavel": "ECC + RF + Médias e SDs", "caminho_pkl": "./data/probas_results_ECC_RF.pkl", "coluna_probas": "probas_instrument_ecc"},
        {"nome_amigavel": "ECC (XGB) + Médias e SDs", "caminho_pkl": "./data/probas_results_ECC_XGB.pkl", "coluna_probas": "probas_instrument_ecc"},
        {"nome_amigavel": "ECC + XGB + Médias e SDs (PCA)", "caminho_pkl": "./data/probas_results_PCA.pkl", "coluna_probas": "probas_instrument_ecc"},
        {"nome_amigavel": "Lightweight CNN", "caminho_pkl": "./data/probas_results_CNN.pkl", "coluna_probas": "pred_cnn"},
        {"nome_amigavel": "CNN 14", "caminho_pkl": "./data/probas_results_CNN14.pkl", "coluna_probas": "pred_cnn"},
        {"nome_amigavel": "ResNet 50", "caminho_pkl": "./data/probas_results_RESNET.pkl", "coluna_probas": "pred_cnn"},
        {"nome_amigavel": "ECC Binário", "caminho_pkl": "./data/rot_unico_ecc.pkl", "coluna_probas": "probas_instrument_ecc"},
        {"nome_amigavel": "DTE Binário", "caminho_pkl": "./data/rot_unico_dte.pkl", "coluna_probas": "probas_instrument_dte"},
    ]
    
    # Dicionários para armazenar resultados
    resultados_gerais_tabela = []
    metricas_por_fold_todos_algos = defaultdict(lambda: defaultdict(list))
    relacao_limiar_f1_por_algo = {}
    
    # --- Processamento de Cada Algoritmo ---
    
    for info in algoritmos_info:
        nome_amigavel = info["nome_amigavel"]
        caminho_pkl = info["caminho_pkl"]
        coluna_probas = info["coluna_probas"]
        eh_caso_especial = nome_amigavel in ["ECC Binário", "DTE Binário"]
        
        print(f"\n\n===== PROCESSANDO ALGORITMO: {nome_amigavel} =====")

        try:
            with open(caminho_pkl, "rb") as f:
                dfs = pickle.load(f)
            if isinstance(dfs, pd.DataFrame):
                dfs = [dfs]
            elif not isinstance(dfs, list):
                print(f"Formato de dados inesperado em {caminho_pkl}. Pulando.")
                continue
        except FileNotFoundError:
            print(f"ARQUIVO NÃO ENCONTRADO: {caminho_pkl}. Pulando.")
            continue
        except Exception as e:
            print(f"Erro ao carregar {caminho_pkl}: {e}. Pulando.")
            continue

        if not dfs:
            print(f"Nenhum fold encontrado em {caminho_pkl}. Pulando.")
            continue

        # Encontrar o melhor limiar e obter métricas por fold associadas
        melhor_limiar, melhor_f1_medio, f1s_por_fold, subsets_por_fold, rel_lim_f1 = encontrar_melhor_limiar(dfs, coluna_probas, eh_caso_especial)
        relacao_limiar_f1_por_algo[nome_amigavel] = rel_lim_f1
        print(f"Melhor limiar encontrado: {melhor_limiar:.2f} (com F1-macro médio de {melhor_f1_medio:.4f})")
        
        # Coletar dados de PR-Curve e outras métricas
        precisions_folds, recalls_folds, avg_precisions_folds = [], [], []
        y_test_todos_folds, y_pred_melhor_limiar, y_pred_limiar_05 = [], [], []

        for df_fold in dfs:
            if 'instrument_label' not in df_fold.columns:
                avg_precisions_folds.append({'micro': np.nan})
                continue

            y_test_i = np.array(df_fold['instrument_label'].tolist())
            y_probs_i = np.array(df_fold[coluna_probas].tolist())
            y_test_todos_folds.append(y_test_i)
            
            # Calcular e guardar dados da curva PR
            prec, rec, _, avg_prec = calcular_curvas_precision_recall(y_test_i, y_probs_i, len(lista_instrumentos))
            precisions_folds.append(prec)
            recalls_folds.append(rec)
            avg_precisions_folds.append(avg_prec)
            
            # Gerar predições para métricas globais
            y_pred_melhor_limiar.append(gerar_predicoes(y_probs_i, melhor_limiar, eh_caso_especial))
            y_pred_limiar_05.append(gerar_predicoes(y_probs_i, 0.5, eh_caso_especial))
            
        # Armazenar métricas por fold para o teste de Wilcoxon
        metricas_por_fold_todos_algos[nome_amigavel]['f1_macro'].extend(f1s_por_fold)
        metricas_por_fold_todos_algos[nome_amigavel]['subset_accuracy'].extend(subsets_por_fold)
        metricas_por_fold_todos_algos[nome_amigavel]['auc_pr_micro'].extend([d.get('micro', np.nan) for d in avg_precisions_folds])

        # Calcular e plotar curva PR média
        n_folds = len(dfs)
        prec_medias = {k: v / n_folds for k, v in soma_dicts_com_ajuste(precisions_folds).items()}
        rec_medias = {k: v / n_folds for k, v in soma_dicts_com_ajuste(recalls_folds).items()}
        avg_prec_medias = {k: v / n_folds for k, v in soma_dicts_com_ajuste(avg_precisions_folds).items()}

        plotar_curvas_precision_recall(prec_medias, rec_medias, avg_prec_medias, len(lista_instrumentos), nome_amigavel, lista_instrumentos)
        
        # Calcular métricas concatenadas (globais)
        y_test_concat = np.vstack(y_test_todos_folds)
        
        y_pred_concat_melhor = np.vstack(y_pred_melhor_limiar)
        f1_macro_melhor_thr = f1_score(y_test_concat, y_pred_concat_melhor, average='macro', zero_division=0)
        subset_acc_melhor_thr = accuracy_score(y_test_concat, y_pred_concat_melhor)

        y_pred_concat_05 = np.vstack(y_pred_limiar_05)
        f1_macro_05_thr = f1_score(y_test_concat, y_pred_concat_05, average='macro', zero_division=0)
        subset_acc_05_thr = accuracy_score(y_test_concat, y_pred_concat_05)

        auc_pr_micro_medio = avg_prec_medias.get('micro', float('nan'))
        
        print(f"\n--- Resumo das Métricas para {nome_amigavel} ---")
        print(f"AUC-PR (micro average) médio: {auc_pr_micro_medio:.4f}")
        print(f"F1-score macro (Melhor Thr={melhor_limiar:.2f}): {f1_macro_melhor_thr:.4f}")
        print(f"Subset Accuracy (Melhor Thr): {subset_acc_melhor_thr:.4f}")
        print(f"F1-score macro (Thr=0.5): {f1_macro_05_thr:.4f}")
        print(f"Subset Accuracy (Thr=0.5): {subset_acc_05_thr:.4f}")
        
        resultados_gerais_tabela.append({
            "modelo": nome_amigavel,
            "F1-Macro (Melhor Thr)": f1_macro_melhor_thr,
            "F1-Macro (Thr 0.5)": f1_macro_05_thr,
            "Melhor Limiar": melhor_limiar,
            "AUC-PR": auc_pr_micro_medio,
            "Subset Acc (Melhor Thr)": subset_acc_melhor_thr,
            "Subset Acc (Thr 0.5)": subset_acc_05_thr
        })

    # --- Apresentação Final dos Resultados ---

    print("\n\n===== TABELA DE RESUMO DOS RESULTADOS =====")
    df_resumo = pd.DataFrame(resultados_gerais_tabela)
    colunas_ordenadas = ["modelo", "F1-Macro (Melhor Thr)", "Subset Acc (Melhor Thr)", 
                         "F1-Macro (Thr 0.5)", "Subset Acc (Thr 0.5)", "Melhor Limiar", "AUC-PR"]
    print(df_resumo[colunas_ordenadas].to_string(index=False))

    # --- Testes de Significância Estatística (Wilcoxon) ---
    print("\n\n===== TESTES DE SIGNIFICÂNCIA ESTATÍSTICA (WILCOXON) =====")
    print("Comparando pares de algoritmos com base em suas métricas por fold.")
    
    nomes_algos_validos = list(metricas_por_fold_todos_algos.keys())

    if not nomes_algos_validos:
        print("Nenhum algoritmo com dados de fold suficientes para realizar os testes.")
    else:
        # F1-Macro
        f1_scores_wilcoxon = {k: v['f1_macro'] for k, v in metricas_por_fold_todos_algos.items()}
        matriz_sig_f1 = criar_matriz_significancia(f1_scores_wilcoxon, nomes_algos_validos)
        plotar_heatmap_significancia(matriz_sig_f1, titulo="Heatmap de Significância (F1-Macro)")
        
        # AUC-PR Micro
        auc_scores_wilcoxon = {k: v['auc_pr_micro'] for k, v in metricas_por_fold_todos_algos.items()}
        matriz_sig_auc = criar_matriz_significancia(auc_scores_wilcoxon, nomes_algos_validos)
        plotar_heatmap_significancia(matriz_sig_auc, titulo="Heatmap de Significância (AUC-PR Micro)")
        
        # Subset Accuracy
        subset_scores_wilcoxon = {k: v['subset_accuracy'] for k, v in metricas_por_fold_todos_algos.items()}
        matriz_sig_subset = criar_matriz_significancia(subset_scores_wilcoxon, nomes_algos_validos)
        plotar_heatmap_significancia(matriz_sig_subset, titulo="Heatmap de Significância (Subset Accuracy)")

    # --- Gráfico Final: Limiar vs. F1-score ---
    plt.figure(figsize=(12, 7))
    
    algoritmos_para_plotar = {
        "ECC (XGB) + Médias e SDs": "crimson",
        "DTE + Médias e SDs": "indigo",
        "ECC + RF + Médias e SDs": "darkgreen"
    }
    
    for nome_algo, valores in relacao_limiar_f1_por_algo.items():
        if nome_algo in algoritmos_para_plotar:
            limiares, f1_scores = zip(*valores)
            cor = algoritmos_para_plotar[nome_algo]
            
            plt.plot(limiares, f1_scores, label=nome_algo, color=cor, linewidth=2.5)
            
            melhor_idx = np.nanargmax(f1_scores)
            plt.plot(limiares[melhor_idx], f1_scores[melhor_idx], 'o', color='gold', 
                     markersize=10, markeredgecolor='black', 
                     label=f'Melhor F1 ({nome_algo}) - Limiar: {limiares[melhor_idx]:.2f}')

    plt.xlabel("Limiar de Binarização")
    plt.ylabel("F1-Macro Médio (nos folds)")
    plt.title("F1-Macro vs. Limiar de Binarização", fontweight='bold')
    plt.legend()
    plt.xlim([0, 1])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("./graphs/limiares_vs_f1_otimizado.pdf")
    plt.show()