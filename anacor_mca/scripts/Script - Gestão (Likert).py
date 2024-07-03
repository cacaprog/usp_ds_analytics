# -*- coding: utf-8 -*-

# Análise de Correspondência Simples e Múltipla
# MBA em Data Science e Analytics USP ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Instalando os pacotes

! pip install pandas
! pip install numpy
! pip install scipy
! pip install plotly
! pip install seaborn
! pip install matplotlib
! pip install statsmodels
! pip install prince

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import prince
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go

#%% Análise de Correspondência Simples (ANACOR)

# Importando o banco de dados

gestao = pd.read_excel("gestao_municipal.xlsx")
# Fonte: Fávero e Belfiore (2024, Capítulo 11)

## Ao longo de 3 anos, foi aplicado questionário com a seguinte afirmação:
    ## Estou satisfeito com a gestão do atual prefeito!

## Respostas em escala likert de 5 pontos

#%% Informações descritivas sobre as variáveis

print(gestao['avaliacao'].value_counts())
print(gestao['ano'].value_counts())

#%% Criando a tabela de contingência

tabela = pd.crosstab(gestao["avaliacao"], gestao["ano"])

print(tabela)

# Nota: sempre manter o nome deste objeto como "tabela" para uso posterior!

#%% Analisando a significância estatística da associação (teste qui²)

teste_qui2 = chi2_contingency(tabela)

print(f"estatística qui²: {round(teste_qui2[0], 2)}")
print(f"p-valor da estatística: {round(teste_qui2[1], 4)}")
print(f"graus de liberdade: {teste_qui2[2]}")

#%% Mapa de calor dos resíduos padronizados ajustados

# Tabela de contingência

tab_cont = sm.stats.Table(tabela)

# Gráfico dos resíduos padronizados ajustados

fig = go.Figure()

maxz = np.max(tab_cont.standardized_resids)+0.1
minz = np.min(tab_cont.standardized_resids)-0.1

colorscale = ['lightgreen' if i>1.96 else '#FAF9F6' for i in np.arange(minz,maxz,0.01)]

fig.add_trace(
    go.Heatmap(
        x = tab_cont.standardized_resids.columns,
        y = tab_cont.standardized_resids.index,
        z = np.array(tab_cont.standardized_resids),
        text=tab_cont.standardized_resids.values,
        texttemplate='%{text:.2f}',
        showscale=False,
        colorscale=colorscale))

fig.update_layout(
    title='Resíduos Padronizados Ajustados',
    height = 600,
    width = 600)

fig.show()

#%% Elaborando a ANACOR

# Na função, o input é a tabela de contingência criada antes!

ca = prince.CA().fit(tabela)

#%% Obtendo os eigenvalues

tabela_autovalores = ca.eigenvalues_summary

print(tabela_autovalores)

# São gerados 'm' autovalores: m = mín(I-1,J-1)

#%% Obtendo a inércia principal total

# É a soma dos eigenvalues (também é a divisão: estat. qui² / N)
# Quanto maior a inércia principal total, maior é a associação entre categorias

print(ca.total_inertia_)

#%% Obtendo as coordenadas do mapa perceptual

# Coordenadas da variável em linha
print(ca.row_coordinates(tabela))

# Coordenadas da variável em coluna
print(ca.column_coordinates(tabela))

#%% Plotando o mapa percentual da Anacor

chart_df_row = pd.DataFrame({'var_row': tabela.index,
                             'x_row':ca.row_coordinates(tabela)[0].values,
                             'y_row': ca.row_coordinates(tabela)[1].values})

chart_df_col = pd.DataFrame({'var_col': tabela.columns,
                             'x_col':ca.column_coordinates(tabela)[0].values,
                             'y_col': ca.column_coordinates(tabela)[1].values})

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.03, point['y'] - 0.02, point['val'], fontsize=6)

label_point(x = chart_df_col['x_col'],
            y = chart_df_col['y_col'],
            val = chart_df_col['var_col'],
            ax = plt.gca())

label_point(x = chart_df_row['x_row'],
            y = chart_df_row['y_row'],
            val = chart_df_row['var_row'],
            ax = plt.gca()) 

sns.scatterplot(data=chart_df_row, x='x_row', y='y_row', s=20)
sns.scatterplot(data=chart_df_col, x='x_col', y='y_col', s=20)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.axhline(y=0, color='lightgrey', ls='--', linewidth=0.8)
plt.axvline(x=0, color='lightgrey', ls='--', linewidth=0.8)
plt.tick_params(size=2, labelsize=6)
plt.title("Mapa Perceptual - Anacor", fontsize=12)
plt.xlabel(f"Dim. 1: {tabela_autovalores.iloc[0,1]} da inércia", fontsize=8)
plt.ylabel(f"Dim. 2: {tabela_autovalores.iloc[1,1]} da inércia", fontsize=8)
plt.show()

#%% Obtendo as coordenadas das observações

# Identificando as variáveis em linha e em coluna
coord_obs = gestao.rename(columns={'avaliacao':'var_row',
                                   'ano':'var_col'})

# Unindo as coordenadas das categorias ao DataFrame
coord_obs = pd.merge(coord_obs, chart_df_row, how='left', on='var_row')
coord_obs = pd.merge(coord_obs, chart_df_col, how='left', on='var_col')

# Calculando as coordenadas médias das observações (média de suas categorias)
coord_obs['x_obs'] = coord_obs[['x_row','x_col']].mean(axis=1)
coord_obs['y_obs'] = coord_obs[['y_row','y_col']].mean(axis=1)

#%% FIM!