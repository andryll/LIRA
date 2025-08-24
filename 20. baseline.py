
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from sklearn.metrics import f1_score, classification_report, hamming_loss, accuracy_score, precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import statistics
from collections import defaultdict

# Criando baselines fracos com base na média e na moda

def get_precision_recall_curves(y_test, y_probs, n_classes):
    
    threshold = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(y_test[:, i], y_probs[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_probs[:, i])
        
    precision["micro"], recall["micro"], threshold["micro"] = precision_recall_curve(y_test.ravel(), y_probs.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_probs, average = 'micro')
    
    return precision, recall, threshold, average_precision

#########################################################################################


def plot_precision_recall_curves(precision, recall, average_precision, n_classes):
    
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue',
                      'teal', 'powderblue', 'darkorchid', 'firebrick',
                      'palegreen', 'gold', 'goldenrod', 'crimson', 'violet',
                      'slategray', 'mediumvioletred', 'darkslateblue',
                      'coral', 'paleturquoise', 'aquamarine', 'lime'])
    lw = 2
    
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], lw=lw, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.4f}'.format(average_precision['micro']))
    plt.legend(loc="lower left")
    plt.show()
    
    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for Each Class')
    plt.show()
    


if __name__ == "__main__":

        df_path = "./data/openmic-2018-labels-filtrado.pkl"
        
        # Manipulando o dataset para separar conjuntos de treino e teste
        df = pd.read_pickle(df_path)
        
        df = df.drop(columns=['families'])

        mlb = MultiLabelBinarizer()

        df_instruments_binarized = pd.DataFrame(
            mlb.fit_transform(df['instrument']),
            columns=mlb.classes_,
            index=df.index
        )
            
        # Combinação Mais Frequente Repetida
        
        row_tuples = [tuple(x) for x in df_instruments_binarized.values]
        most_common_tuple, frequency = Counter(row_tuples).most_common(1)[0]
        most_frequent_combination = pd.Series(most_common_tuple, index=df_instruments_binarized.columns)

        print(f"Combinação de features mais frequente (ocorre {frequency} vezes):")
        print(most_frequent_combination)
        print("-" * 30)
        
        df_frequent_combination_repeated = pd.DataFrame(
            [most_frequent_combination.values] * len(df), 
            columns=df_instruments_binarized.columns, 
            index=df.index 
            )

        instrument_means = pd.Series(df_instruments_binarized.mean(), index=df_instruments_binarized.columns)
        
        df_instrument_means = pd.DataFrame(
            [instrument_means.values] * len(df), 
            columns=df_instruments_binarized.columns, 
            index=df.index 
            )

        print("Segundo DataFrame Novo (Média de Cada Instrumento):")
        print(df_instrument_means)
        print("-" * 30)
        
        y_true_binarized = df_instruments_binarized.values
        
        y_pred_frequent = df_frequent_combination_repeated.values
        y_probs_frequent = y_pred_frequent.astype(float)
        
        precision, recall, threshold, avg_prec = get_precision_recall_curves(y_true_binarized, y_probs_frequent, len((mlb.classes_)))

        plot_precision_recall_curves(precision, recall, avg_prec, len(mlb.classes_))
        
        print(f"F1-Score para a moda: {f1_score(y_true_binarized, y_probs_frequent, average='macro')}")
        print(f"Subset Accuracy para a moda: {accuracy_score(y_true_binarized, y_probs_frequent)}")
        
        
        y_probs_means = df_instrument_means.values
        y_pred_means = (y_probs_means > 0.5).astype(int)
        
        precision, recall, threshold, avg_prec = get_precision_recall_curves(y_true_binarized, y_probs_means, len((mlb.classes_)))

        plot_precision_recall_curves(precision, recall, avg_prec, len(mlb.classes_))
        
        print(f"F1-Score para a média: {f1_score(y_true_binarized, y_pred_means, average='macro')}")
        print(f"Subset Accuracy para a média: {accuracy_score(y_true_binarized, y_pred_means)}")
        
    
        print("Finalizando programa")