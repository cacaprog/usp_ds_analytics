# -*- coding: utf-8 -*-

## Exercícios de Técnicas Supervisionadas de ML (MBA DSA USP/ESALQ)
# Prof. Dr. Wilson Tarantin Jr.

#%% Instalando os pacotes necessários

!pip install pandas
!pip install numpy
!pip install statsmodels
!pip install matplotlib
!pip install -U seaborn
!pip install pingouin
!pip install statstests
!pip install scipy
!pip install scikit-learn

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
from statstests.process import stepwise
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score

#%% Importando o banco de dados

demissao = pd.read_excel('recursos_humanos.xlsx')
## Fonte: adaptado de https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

demissao.info()

#%% Estatísticas descritivas

# Variáveis métricas
demissao[['idade', 'dist_residencia', 'salario']].describe()
demissao[['qtd_treinamentos', 'anos_empresa', 'anos_promo']].describe()

# Variáveis categóricas
demissao['demissao'].value_counts().sort_index()
demissao['satisfacao_amb'].value_counts().sort_index()
demissao['sexo'].value_counts().sort_index()
demissao['satisfacao_cargo'].value_counts().sort_index()
demissao['estado_civil'].value_counts().sort_index()
demissao['hora_extra'].value_counts().sort_index()

#%% Recodificando a variável dependente 'demissão'

# Codificando evento = 1; não evento = 0
demissao.loc[demissao['demissao']=='sim', 'demissao'] = 1
demissao.loc[demissao['demissao']=='nao', 'demissao'] = 0

# Transformando em variável numérica
demissao['demissao'] = demissao['demissao'].astype('int')

#%% Criando as n-1 dummies das variáveis explicativas categóricas

demissao_dummies = pd.get_dummies(demissao,
                                  columns=['satisfacao_amb',
                                           'sexo',
                                           'satisfacao_cargo',
                                           'estado_civil',
                                           'hora_extra'],
                                  dtype=int,
                                  drop_first=True)

#%% Criando o texto da fórmula

def texto_formula(df, var_dependente, excluir_cols):
    variaveis = list(df.columns.values)
    variaveis.remove(var_dependente)
    for col in excluir_cols:
        variaveis.remove(col)
    return var_dependente + ' ~ ' + ' + '.join(variaveis)

texto_regressao = texto_formula(demissao_dummies, 'demissao', '')
# 1º argumento: banco de dados
# 2º argumento: variável dependente
# 3º argumento: variáveis a serem excluídas (se houver, inserir como lista)

#%% Regressão Logística Binária

# Estimando o modelo
modelo_demissao = sm.Logit.from_formula(texto_regressao,
                                        demissao_dummies).fit()

# Analisando os resultados
modelo_demissao.summary()

## Algumas variáveis não apresentam significância estatística ao nível de 5%

#%% Obtendo o modelo após o procedimento de stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

modelo_stepwise = stepwise(modelo_demissao, pvalue_limit=0.05)

#%% Armazenando os valores previstos para amostra

demissao_dummies['previsto'] = modelo_stepwise.predict()

## Refere-se à probabilidade de ocorrência do evento (demissão = 1)

#%% Matriz de confusão (definição da função)

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True (Observado)')
    plt.ylabel('Classified (Predito Modelo)')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

#%% Matriz de confusão 

# Cutoff = 0.30
matriz_confusao(observado=demissao_dummies['demissao'],
                predicts=demissao_dummies['previsto'], 
                cutoff=0.30)

# Cutoff = 0.35
matriz_confusao(observado=demissao_dummies['demissao'],
                predicts=demissao_dummies['previsto'], 
                cutoff=0.35)

# A avaliação por meio da matriz de confusão depende do cutoff escolhido

#%% Análise da curva ROC

# Parametrizando a função da curva ROC (real vs. previsto)
fpr, tpr, thresholds = roc_curve(demissao_dummies['demissao'], demissao_dummies['previsto'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

# Plotando a curva ROC
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%% Realizando predições

# Qual é a probabilidade média esperada de um funcionário  
# com as seguintes características pedir demissão?

prob_pred = modelo_stepwise.predict(pd.DataFrame({'satisfacao_amb_baixa':[1],
                                                  'satisfacao_cargo_baixo': [0],
                                                  'satisfacao_cargo_muito_alta': [1],
                                                  'estado_civil_solteiro': [0],
                                                  'hora_extra_sim': [0],
                                                  'idade': [30],
                                                  'dist_residencia': [5],
                                                  'salario': [3000],
                                                  'qtd_treinamentos': [2],
                                                  'anos_empresa': [6],
                                                  'anos_promo': [2]}))

# O resultado mostra a probabilidade média estimada de ocorrência do evento
print(f"Probabilidade Predita: {round(prob_pred[0]*100, 2)}%")

#%% Analisando a odds ratio das variáveis explicativas

# Coeficientes betas estimados pelo modelo stepwise
modelo_stepwise.params[1:]

###############################################################################

# Exemplo: odds ratio da variável 'estado_civil_solteiro'
np.exp(1.019344)

# Em média, mantidas as demais condições constates, a chance de um funcionário
# pedir demissão é multiplicada por um fator de 2.77 ao ser solteiro 
# ao invés de ser casado ou divorciado

# Portanto, em média, a chance é 177% maior

# Para detalhar o resultado, note que a chance = p / (1 - p)

# Não solteiro (portanto, ser casado ou divorciado)
0.116 / (1 - 0.116)

# Solteiro (realizar o predict para solteiro = 1 sem alterar outros)
0.2667 / (1 - 0.2667)

###############################################################################

# Exemplo: odds ratio da variável 'qtd_treinamentos'
np.exp(-0.131297)

# Em média, mantidas as demais condições constates, a chance de um funcionário
# pedir demissão é multiplicada por um fator de 0.8769
# ao receber 1 treinamento a mais

# Portanto, em média, a chance é 12,31% menor

# Para detalhar o resultado, note que a chance = p / (1 - p)

# qtd_treinamentos = 2
0.116 / (1 - 0.116)

# qtd_treinamentos = 3 (realizar o novo predict sem alterar outros)
0.1032 / (1 - 0.1032)

#%% Fim!