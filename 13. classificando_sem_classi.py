
import pickle
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_recall_curve)

# --- Constantes e Configurações ---
NOME_ARQUIVO_CACHE = "dados_dos_folds.pkl"
CORES_GRAFICOS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
LISTA_INSTRUMENTOS_PT = [
    'clarinete', 'flauta', 'trompete', 'saxofone', 'vocal', 'sanfona', 'ukulelê',
    'percussão com baquetas', 'piano', 'violão', 'bandolin', 'banjo', 'sintetizador',
    'trombone', 'orgão', 'bateria', 'baixo', 'pratos', 'violoncelo', 'violino'
]
N_CLASSES = len(LISTA_INSTRUMENTOS_PT)
ALPHA_WILCOXON = 0.05

# --- Funções de Métricas e Visualização ---

def calcular_curvas_pr(y_test: np.ndarray, y_probs: np.ndarray) -> Tuple[Dict, Dict, Dict]:
    """Calcula as curvas Precision-Recall para cada classe e a média micro."""
    precisao, recall, avg_precision = {}, {}, {}
    for i in range(N_CLASSES):
        precisao[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_probs[:, i])
        avg_precision[i] = average_precision_score(y_test[:, i], y_probs[:, i])
    
    precisao["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_probs.ravel())
    avg_precision["micro"] = average_precision_score(y_test, y_probs, average='micro')
    return precisao, recall, avg_precision

def plotar_curvas_pr(precisao: Dict, recall: Dict, avg_precision: Dict, nome_amigavel: str):
    """Plota as curvas Precision-Recall"""
    cores = cycle(CORES_GRAFICOS)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if "micro" in recall:
        ax.plot(recall["micro"], precisao["micro"], color='gold', lw=3, linestyle='--',
                label=f'Curva Micro-Média (área = {avg_precision.get("micro", 0):.3f})')
                
    for i, cor in zip(range(N_CLASSES), cores):
        if i in recall:
            ax.plot(recall[i], precisao[i], color=cor, lw=2,
                    label=f'Curva {LISTA_INSTRUMENTOS_PT[i]} (área = {avg_precision.get(i, 0):.3f})')
            
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='Recall', ylabel='Precisão')
    ax.set_title(f'Curva Precision-Recall para cada classe\n({nome_amigavel})', fontweight='bold')
    ax.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def criar_matriz_significancia(scores_por_fold: Dict, nomes_algos: List[str]) -> pd.DataFrame:
    """Cria uma matriz de significância estatística usando o teste de Wilcoxon."""
    matriz = pd.DataFrame(index=nomes_algos, columns=nomes_algos, dtype=str)
    for i, nome1 in enumerate(nomes_algos):
        for j, nome2 in enumerate(nomes_algos):
            if i == j:
                matriz.loc[nome1, nome2] = "-"
                continue
            scores1, scores2 = scores_por_fold.get(nome1, []), scores_por_fold.get(nome2, [])
            if len(scores1) != len(scores2) or not scores1:
                matriz.loc[nome1, nome2] = "dados_inválidos"
                continue
            try:
                _, p_valor = wilcoxon(scores1, scores2, zero_method='pratt', alternative='two-sided')
                if p_valor < ALPHA_WILCOXON:
                    med1, med2 = np.median(scores1), np.median(scores2)
                    resultado = '+' if med1 > med2 else '-' if med1 < med2 else '~eq'
                    matriz.loc[nome1, nome2] = f"{resultado} (p={p_valor:.3f})"
                else:
                    matriz.loc[nome1, nome2] = f"~ (p={p_valor:.3f})"
            except ValueError:
                matriz.loc[nome1, nome2] = "erro_estatístico"
    return matriz

def plotar_heatmap_significancia(matriz_sig: pd.DataFrame, titulo: str):
    """Plota um heatmap visual para a matriz de significância."""
    n = len(matriz_sig)
    mascara_bool = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            val = matriz_sig.iat[i, j]
            if isinstance(val, str) and (val.startswith("+") or val.startswith("-")):
                mascara_bool[i, j] = True

    mascara_triangular = np.tril(np.ones_like(mascara_bool, dtype=bool))
    cmap = mcolors.ListedColormap(['#ff7f7f', '#90ee90']) # Vermelho claro, Verde claro
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(np.ma.masked_array(mascara_bool.astype(int), mask=mascara_triangular), cmap=cmap)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(matriz_sig.columns, rotation=45, ha='right')
    ax.set_yticklabels(matriz_sig.index)
    ax.grid(False)
    
    legenda = [Patch(facecolor='#90ee90', label='Diferença Significativa'),
               Patch(facecolor='#ff7f7f', label='Não Significativo')]
    ax.legend(handles=legenda, loc='best')
    ax.set_title(titulo, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- Funções de Processamento e Avaliação ---

def encontrar_melhor_limiar(dfs: List[pd.DataFrame], coluna_probas: str) -> Tuple:
    """Otimiza o limiar de decisão buscando o maior F1-Macro médio entre os folds."""
    melhor_f1, melhor_limiar = -1, 0.5
    f1_no_melhor_limiar, ss_acc_no_melhor_limiar = [], []
    relacao_limiar_f1 = []

    for limiar in np.arange(0.0, 1.01, 0.01):
        f1_temp, ss_acc_temp = [], []
        for df_fold in dfs:
            y_test = np.array(df_fold['instrument_label'].tolist())
            y_probs = np.array(df_fold[coluna_probas].tolist())
            y_pred = (y_probs >= limiar).astype(int)
            
            f1_temp.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            ss_acc_temp.append(accuracy_score(y_test, y_pred))
            
        f1_medio = np.nanmean(f1_temp)
        relacao_limiar_f1.append((limiar, f1_medio))
        if f1_medio > melhor_f1:
            melhor_f1, melhor_limiar = f1_medio, limiar
            f1_no_melhor_limiar, ss_acc_no_melhor_limiar = f1_temp.copy(), ss_acc_temp.copy()
            
    return melhor_limiar, f1_no_melhor_limiar, ss_acc_no_melhor_limiar, relacao_limiar_f1

def processar_modelo_base(info: Dict, dfs: List[pd.DataFrame]) -> Dict:
    """Processa um modelo base, otimiza seu limiar e calcula todas as métricas."""
    print(f"\n--- Processando Modelo Base: {info['nome_amigavel']} ---")
    
    # 1. Otimizar limiar
    melhor_limiar, f1_per_fold, ss_acc_per_fold, rel_lim_f1 = encontrar_melhor_limiar(dfs, info['coluna_probas'])
    print(f"Melhor limiar encontrado: {melhor_limiar:.2f}")

    # 2. Calcular AUC-PR por fold
    auc_pr_per_fold = [calcular_curvas_pr(np.array(df['instrument_label'].tolist()), 
                                          np.array(df[info['coluna_probas']].tolist()))[2].get("micro", np.nan)
                       for df in dfs]

    # 3. Concatenar resultados para métricas globais
    all_y_test = np.vstack([np.array(df['instrument_label'].tolist()) for df in dfs])
    all_y_probs = np.vstack([np.array(df[info['coluna_probas']].tolist()) for df in dfs])
    
    y_pred_best = (all_y_probs >= melhor_limiar).astype(int)
    y_pred_05 = (all_y_probs >= 0.5).astype(int)

    return {
        "nome_amigavel": info['nome_amigavel'],
        "metricas_por_fold": {'f1_macro': f1_per_fold, 'subset_accuracy': ss_acc_per_fold, 'auc_pr_micro': auc_pr_per_fold},
        "relacao_limiar_f1": rel_lim_f1,
        "resultados_tabela": {
            "modelo": info['nome_amigavel'],
            "AUC-PR (micro)": np.nanmean(auc_pr_per_fold),
            "F1-Macro (Best Thr)": f1_score(all_y_test, y_pred_best, average='macro', zero_division=0),
            "Subset Acc (Best Thr)": accuracy_score(all_y_test, y_pred_best),
            "F1-Macro (Thr 0.5)": f1_score(all_y_test, y_pred_05, average='macro', zero_division=0),
            "Subset Acc (Thr 0.5)": accuracy_score(all_y_test, y_pred_05),
            "Best Threshold": melhor_limiar,
        }
    }

def processar_modelo_simulado(sim_info: Dict, cache_dfs: Dict, cache_metricas: Dict) -> Dict:
    """Processa um modelo simulado a partir de um modelo base, aplicando regras específicas."""
    nome_simulado = sim_info['nome_amigavel']
    nome_fonte = sim_info['source_model']
    print(f"\n--- Processando Modelo Simulado: {nome_simulado} (a partir de {nome_fonte}) ---")
    
    if nome_fonte not in cache_dfs:
        print(f"AVISO: Dados fonte '{nome_fonte}' não encontrados. Pulando simulação.")
        return None

    # 1. Aplicar lógica de simulação
    f1_sim_per_fold, ss_acc_sim_per_fold = [], []
    all_y_test, all_y_pred_sim = [], []
    
    for df_fold in cache_dfs[nome_fonte]:
        y_test = np.array(df_fold['instrument_label'].tolist())
        y_probs = np.array(df_fold[sim_info['coluna_probas']].tolist())
        
        y_pred = (y_probs >= 0.5).astype(int)
        mask_sem_predicao = np.sum(y_pred, axis=1) == 0
        if np.any(mask_sem_predicao):
            # Atribui a classe de maior probabilidade onde nenhuma foi prevista
            indices_max = np.argmax(y_probs[mask_sem_predicao], axis=1)
            y_pred[mask_sem_predicao, indices_max] = 1
            
        all_y_test.append(y_test)
        all_y_pred_sim.append(y_pred)
        f1_sim_per_fold.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        ss_acc_sim_per_fold.append(accuracy_score(y_test, y_pred))

    # 2. Calcular métricas globais e reutilizar AUC-PR da fonte
    y_test_cat, y_pred_cat = np.vstack(all_y_test), np.vstack(all_y_pred_sim)
    f1_global, ss_acc_global = f1_score(y_test_cat, y_pred_cat, average='macro', zero_division=0), accuracy_score(y_test_cat, y_pred_cat)

    return {
        "nome_amigavel": nome_simulado,
        "metricas_por_fold": {'f1_macro': f1_sim_per_fold, 'subset_accuracy': ss_acc_sim_per_fold, 
                              'auc_pr_micro': cache_metricas[nome_fonte]['auc_pr_micro']},
        "resultados_tabela": {
            "modelo": nome_simulado,
            "AUC-PR (micro)": np.nanmean(cache_metricas[nome_fonte]['auc_pr_micro']),
            "F1-Macro (Best Thr)": f1_global, "F1-Macro (Thr 0.5)": f1_global,
            "Subset Acc (Best Thr)": ss_acc_global, "Subset Acc (Thr 0.5)": ss_acc_global,
            "Best Threshold": 0.5,
        }
    }

if __name__ == "__main__":
    
    # --- Configuração dos Modelos ---
    modelos_base_info = [
        {"nome_amigavel": "DTE + Médias e SDs", "caminho_pkl": "./data/results_dte.pkl", "coluna_probas": "probas_instrument_dte"},
        {"nome_amigavel": "ECC + RF + Médias e SDs", "caminho_pkl": "./data/probas_results_ECC_RF.pkl", "coluna_probas": "probas_instrument_ecc"},
        {"nome_amigavel": "ECC + XGB + Médias e SDs", "caminho_pkl": "./data/probas_results_ECC_XGB.pkl", "coluna_probas": "probas_instrument_ecc"},
    ]
    modelos_simulados_info = [
        {"nome_amigavel": "DTE Binário (simulado)", "source_model": "DTE + Médias e SDs", "coluna_probas": "probas_instrument_dte"},
        {"nome_amigavel": "ECC Binário (simulado)", "source_model": "ECC + XGB + Médias e SDs", "coluna_probas": "probas_instrument_ecc"}
    ]

    print("===== ETAPA 1: CARREGANDO DADOS DOS ARQUIVOS PKL =====")
    cache_dfs = {}
    for info in modelos_base_info:
        caminho = Path(info['caminho_pkl'])
        if not caminho.exists():
            print(f"ERRO: Arquivo não encontrado: {caminho}. Pulando.")
            continue
        try:
            with open(caminho, "rb") as f:
                dados = pickle.load(f)
            # Normaliza os dados para sempre ser uma lista de DataFrames
            if isinstance(dados, pd.DataFrame): cache_dfs[info['nome_amigavel']] = [dados]
            elif isinstance(dados, list): cache_dfs[info['nome_amigavel']] = dados
            elif isinstance(dados, dict): cache_dfs[info['nome_amigavel']] = list(dados.values())
            else: print(f"Formato de dados não suportado em {caminho}.")
        except Exception as e:
            print(f"Falha ao carregar {caminho}: {e}")

    print("\n===== ETAPA 2: PROCESSANDO E AVALIANDO OS MODELOS =====")
    resultados_finais = []
    metricas_por_fold_agregadas = {}
    relacao_limiar_f1_por_algo = {}

    # Processa modelos base
    for info in modelos_base_info:
        if info['nome_amigavel'] in cache_dfs:
            resultado = processar_modelo_base(info, cache_dfs[info['nome_amigavel']])
            resultados_finais.append(resultado['resultados_tabela'])
            metricas_por_fold_agregadas[info['nome_amigavel']] = resultado['metricas_por_fold']
            relacao_limiar_f1_por_algo[info['nome_amigavel']] = resultado['relacao_limiar_f1']

    # Processa modelos simulados
    for info in modelos_simulados_info:
        resultado = processar_modelo_simulado(info, cache_dfs, metricas_por_fold_agregadas)
        if resultado:
            resultados_finais.append(resultado['resultados_tabela'])
            metricas_por_fold_agregadas[info['nome_amigavel']] = resultado['metricas_por_fold']

    # Exibir Resultados e Análise Estatística 
    print("\n\n===== ETAPA 3: RESULTADOS FINAIS E ANÁLISE =====")
    df_resumo = pd.DataFrame(resultados_finais).sort_values(by="AUC-PR (micro)", ascending=False)
    colunas_ordem = ["modelo", "AUC-PR (micro)", "F1-Macro (Best Thr)", "Subset Acc (Best Thr)",
                     "F1-Macro (Thr 0.5)", "Subset Acc (Thr 0.5)", "Best Threshold"]
    print(df_resumo[colunas_ordem].to_string(float_format="%.4f", index=False))

    print("\n\n--- TESTES DE SIGNIFICÂNCIA ESTATÍSTICA (WILCOXON) ---")
    nomes_algos = list(metricas_por_fold_agregadas.keys())
    
    for metrica, titulo in [
        ('f1_macro', 'F1-Macro (com melhor limiar)'),
        ('subset_accuracy', 'Subset Accuracy (com melhor limiar)'),
        ('auc_pr_micro', 'AUC-PR (micro)')
    ]:
        scores = {nome: metricas[metrica] for nome, metricas in metricas_por_fold_agregadas.items()}
        matriz_sig = criar_matriz_significancia(scores, nomes_algos)
        print(f"\nMatriz de Significância para {titulo}:\n{matriz_sig.to_string()}")
        plotar_heatmap_significancia(matriz_sig, f"Heatmap de Significância ({titulo})")