import pandas as pd
import matplotlib.pyplot as plt
import os 

path = "./../openmic-2018/openmic-2018-aggregated-labels.csv"


data = pd.read_csv(path)

# Mapeando as famílias
family_mapping = {
    'clarinet': 'woodwinds',
    'flute': 'woodwinds',
    'saxophone': 'woodwinds',
    'trumpet': 'brass',
    'trombone': 'brass',
    'ukulele': 'plucked_strings',
    'guitar': 'plucked_strings',
    'banjo': 'plucked_strings',
    'bass': 'electronic',
    'mandolin': 'plucked_strings',
    'cello': 'bowed_strings',
    'violin': 'bowed_strings',
    'piano': 'pitched_percussion', 
    'drums': 'unpitched_percussion',
    'cymbals': 'unpitched_percussion',
    'mallet_percussion': 'pitched_percussion',
    'accordion': 'aerophones',
    'organ': 'aerophones', 
    'synthesizer': 'electronic',
    'voice': 'vocals'
}

# Criando a coluna 'families' com base no mapeamento feito
data['families'] = data['instrument'].map(family_mapping)

# Contando instrumentos antes da filtragem
instrument_counts_antes_filtro = data['instrument'].value_counts()

# Agrupando instrumentos e famílias
df_instruments = data.groupby('sample_key')['instrument'].apply(list).reset_index()
df_families = data.groupby('sample_key')['families'].apply(lambda x: list(set(x))).reset_index()
df = pd.merge(df_instruments, df_families, on='sample_key')

# Filtrando combinações abaixo de um Threshold
threshold = 10


df['combo_key'] = df['instrument'].apply(lambda x: tuple(sorted(x)))
n_combos1 = df['combo_key'].nunique() 
print(f"Total de combinações únicas de instrumentos ANTES do filtro: {n_combos1}")

combo_counts = df['combo_key'].value_counts()
valid_combos = combo_counts[combo_counts >= threshold].index

# Filtrando Dataframe
df_filtrado = df[df['combo_key'].isin(valid_combos)].reset_index(drop=True)
n_combos2 = df_filtrado['combo_key'].nunique()
print(f"Total de combinações únicas de instrumentos DEPOIS do filtro: {n_combos2}")

df_filtrado = df_filtrado.drop(columns=['combo_key'])


data_filtrada_exploded = df_filtrado.explode('instrument')
data_filtrada_exploded['families'] = data_filtrada_exploded['instrument'].map(family_mapping)


novo_total_de_musicas = df_filtrado.shape[0]
print(f"Número total de músicas (amostras) APÓS a filtragem por combinação: {novo_total_de_musicas}")
print("-" * 70)


instrument_counts_depois_filtro = data_filtrada_exploded['instrument'].value_counts()
instrument_rate_depois_filtro = instrument_counts_depois_filtro / novo_total_de_musicas 

print("Contagem de cada instrumento APÓS a filtragem (considerando ocorrências em músicas filtradas):")
print(instrument_counts_depois_filtro)
print("-" * 70)
print("Taxa de presença de cada instrumento APÓS a filtragem:")
print(instrument_rate_depois_filtro)
print("-" * 70)

all_instrument_names = pd.Index(instrument_counts_antes_filtro.index).union(pd.Index(instrument_counts_depois_filtro.index))

summary_table = pd.DataFrame(index=all_instrument_names)
summary_table['Instrumento'] = summary_table.index 

summary_table['Contagem Antes Filtro'] = instrument_counts_antes_filtro.reindex(all_instrument_names, fill_value=0)
summary_table['Contagem Depois Filtro'] = instrument_counts_depois_filtro.reindex(all_instrument_names, fill_value=0)
summary_table['Taxa Aparição Depois Filtro'] = instrument_rate_depois_filtro.reindex(all_instrument_names, fill_value=0)


summary_table['Percentual Redução (%)'] = 0.0 
mask_antes_gt_0 = summary_table['Contagem Antes Filtro'] > 0

summary_table.loc[mask_antes_gt_0, 'Percentual Redução (%)'] = \
    ((summary_table['Contagem Antes Filtro'][mask_antes_gt_0] - summary_table['Contagem Depois Filtro'][mask_antes_gt_0]) /
     summary_table['Contagem Antes Filtro'][mask_antes_gt_0]) * 100


summary_table.loc[(summary_table['Contagem Antes Filtro'] > 0) & (summary_table['Contagem Depois Filtro'] == 0), 'Percentual Redução (%)'] = 100.0

summary_table = summary_table[['Instrumento', 'Contagem Antes Filtro', 'Contagem Depois Filtro', 'Taxa Aparição Depois Filtro', 'Percentual Redução (%)']]
summary_table = summary_table.sort_values(by='Contagem Depois Filtro', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("TABELA RESUMO POR INSTRUMENTO (APÓS FILTRAGEM POR COMBINAÇÃO)")
print("="*80)

summary_table_display = summary_table.copy()
summary_table_display['Taxa Aparição Depois Filtro'] = summary_table_display['Taxa Aparição Depois Filtro'].map('{:.4f}'.format)
summary_table_display['Percentual Redução (%)'] = summary_table_display['Percentual Redução (%)'].map('{:.2f}%'.format)
print(summary_table_display.to_string()) 
print("="*80 + "\n")


# Plot da presença dos instrumentos
plt.figure(figsize=(12, 6))
instrument_rate_depois_filtro.plot(kind='bar', color='skyblue') 
plt.title('Taxa de Presença dos Instrumentos (Após Filtragem por Combinação)')
plt.xlabel("Instrumento")
plt.ylabel('Taxa de Presença (Nº Ocorrências / Nº Músicas Filtradas)')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

family_counts_filtrado = data_filtrada_exploded['families'].value_counts()
family_rate_filtrado = family_counts_filtrado / novo_total_de_musicas

print("Contagem de cada família de instrumentos APÓS a filtragem:")
print(family_counts_filtrado)
print("-" * 70)
print("Taxa de presença de cada família de instrumentos APÓS a filtragem:")
print(family_rate_filtrado)
print("-" * 70)

plt.figure(figsize=(10, 5))
family_rate_filtrado.plot(kind='bar', color='lightcoral')
plt.title('Taxa de Presença das Famílias de Instrumentos (Após Filtragem por Combinação)')
plt.xlabel("Família de Instrumentos")
plt.ylabel('Taxa de Presença (Nº Ocorrências / Nº Músicas Filtradas)')
plt.xticks(rotation=45, ha="right")
plt.tight_layout() # Adjust layout
plt.show()

# Salvando DF
output_dir = "./data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Salvando DF antes da filtragem
df.to_pickle(path=os.path.join(output_dir, "openmic-2018-labels_original_grouped.pkl"))
# Salvando DF depois da filtragem
df_filtrado.to_pickle(path=os.path.join(output_dir, "openmic-2018-labels_filtered_grouped.pkl"))
# Salvando a tabela de resumo
summary_table.to_csv(os.path.join(output_dir, "openmic-2018-instrument_summary_after_filtering.csv"), index=False)


print(f"DataFrame original agrupado salvo em: {os.path.join(output_dir, 'openmic-2018-labels_original_grouped.pkl')}")
print(f"DataFrame filtrado agrupado salvo em: {os.path.join(output_dir, 'openmic-2018-labels_filtered_grouped.pkl')}")
print(f"Tabela resumo dos instrumentos salva em: {os.path.join(output_dir, 'openmic-2018-instrument_summary_after_filtering.csv')}")
print("-" * 70)

print(f"Resumo da Filtragem por Combinação:")
print(f"Total de combinações únicas de instrumentos ANTES do filtro: {n_combos1}")
print(f"Total de combinações únicas de instrumentos DEPOIS do filtro: {n_combos2}")
if n_combos1 > 0 :
    print(f"Redução no número de combinações: {n_combos1 - n_combos2} ({(n_combos1 - n_combos2) / n_combos1 * 100:.2f}%)")
else:
    print(f"Redução no número de combinações: {n_combos1 - n_combos2} (N/A %)")


original_song_count = df['sample_key'].nunique() 
print(f"Número de músicas (amostras) ANTES do filtro por combinação: {original_song_count}")
print(f"Número de músicas (amostras) APÓS a filtragem por combinação: {novo_total_de_musicas}")
if original_song_count > 0:
    print(f"Redução no número de músicas: {original_song_count - novo_total_de_musicas} ({(original_song_count - novo_total_de_musicas) / original_song_count * 100:.2f}%)")
else:
    print(f"Redução no número de músicas: {original_song_count - novo_total_de_musicas} (N/A %)")

