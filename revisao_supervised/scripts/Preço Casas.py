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

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
from statstests.process import stepwise
from statstests.tests import shapiro_francia
from scipy.stats import boxcox
from scipy.stats import norm
from scipy import stats

#%% Importando o banco de dados

dados = pd.read_excel('preço_casas.xlsx')
## Fonte: adaptado de https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

dados.info()

#%% Estatísticas descritivas dos dados

dados[['preco', 'area_sqm', 'quartos', 'banheiros', 'vagas_garagem']].describe()

dados['rua_principal'].value_counts().sort_index()
dados['quarto_hospedes'].value_counts().sort_index()
dados['porao'].value_counts().sort_index()
dados['aquecimento_agua'].value_counts().sort_index()
dados['ar_condicionado'].value_counts().sort_index()
dados['local_pref'].value_counts().sort_index()
dados['mobilia'].value_counts().sort_index()

#%% Análise do coeficiente de correlação de Pearson entre as variáveis

pg.rcorr(dados[['preco', 'area_sqm', 'quartos', 'banheiros', 'vagas_garagem']],
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Obtendo as dummies de variáveis categóricas

dados = pd.get_dummies(dados,
                       columns = ['rua_principal',
                                  'quarto_hospedes',
                                  'porao',
                                  'aquecimento_agua',
                                  'ar_condicionado',
                                  'local_pref',
                                  'mobilia'],
                       dtype = int,
                       drop_first = True)

#%% Modelo de Regressão Linear Múltipla (MQO)

# Estimação do modelo
reg = sm.OLS.from_formula(formula = 'preco ~ area_sqm + quartos + banheiros + \
                                     rua_principal_sim + quarto_hospedes_sim + \
                                     porao_sim + aquecimento_agua_sim + \
                                     ar_condicionado_sim + vagas_garagem + \
                                     local_pref_sim + mobilia_mobiliado + \
                                     mobilia_sem_mobilia',
                          data=dados).fit()

# Obtenção dos outputs
reg.summary()

#%% Teste de verificação da aderência dos resíduos à normalidade

# Elaboração do teste de Shapiro-Francia
teste_sf = shapiro_francia(reg.resid)
round(teste_sf['p-value'], 5)

# Tomando a decisão por meio do teste de hipóteses

alpha = 0.05 # nível de significância do teste

if teste_sf['p-value'] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

#%% Histograma dos resíduos do modelo OLS

# Parâmetros de referência para a distribuição normal teórica
(mu, std) = norm.fit(reg.resid)

# Criação do gráfico
plt.figure(figsize=(15,10))
plt.hist(reg.resid, bins=35, density=True, alpha=0.7, color='purple')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)
plt.plot(x, p, linewidth=3, color='red')
plt.title('Resíduos do Modelo', fontsize=20)
plt.xlabel('Resíduos', fontsize=22)
plt.ylabel('Frequência', fontsize=22)
plt.show()

#%% Realizando a transformação de Box-Cox na variável dependente

y_box, lmbda = boxcox(dados['preco'])

# Valor obtido para o lambda
print(lmbda)

# Adicionando ao banco de dados
dados['preco_bc'] = y_box

#%% Modelo de regressão com transformação de Box-Cox em Y

# Estimação do modelo
reg_bc = sm.OLS.from_formula(formula = 'preco_bc ~ area_sqm + quartos + banheiros + \
                                        rua_principal_sim + quarto_hospedes_sim + \
                                        porao_sim + aquecimento_agua_sim + \
                                        ar_condicionado_sim + vagas_garagem + \
                                        local_pref_sim + mobilia_mobiliado + \
                                        mobilia_sem_mobilia',
                             data=dados).fit()

# Obtenção dos outputs
reg_bc.summary()

#%% Reavaliando aderência à normalidade dos resíduos do modelo

# Teste de Shapiro-Francia
teste_sf_bc = shapiro_francia(reg_bc.resid)

# Tomando a decisão por meio do teste de hipóteses

alpha = 0.05 # nível de significância do teste

if teste_sf_bc['p-value'] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

#%% Removendo as variáveis que não apresentam significância estatística

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

# Stepwise do modelo
modelo_stepwise_bc = stepwise(reg_bc, pvalue_limit=0.05)

# Teste de Shapiro-Francia
teste_sf_step = shapiro_francia(modelo_stepwise_bc.resid)

#%% Novo histograma dos resíduos do modelo

# Parâmetros de referência para a distribuição normal teórica
(mu_bc, std_bc) = norm.fit(modelo_stepwise_bc.resid)

# Criação do gráfico
plt.figure(figsize=(15,10))
plt.hist(modelo_stepwise_bc.resid, bins=30, density=True, alpha=0.8, color='darkblue')
xmin_bc, xmax_bc = plt.xlim()
x_bc = np.linspace(xmin_bc, xmax_bc, 1000)
p_bc = norm.pdf(x_bc, mu_bc, std_bc)
plt.plot(x_bc, p_bc, linewidth=3, color='red')
plt.title('Resíduos do Modelo Box-Cox', fontsize=20)
plt.xlabel('Resíduos', fontsize=22)
plt.ylabel('Frequência', fontsize=22)
plt.show()

#%% Realizando predições com base no modelo estimado

# Modelo Não Linear (Box-Cox):
valor_pred_bc = modelo_stepwise_bc.predict(pd.DataFrame({'area_sqm':[350],
                                                         'quartos': [3],
                                                         'banheiros': [3],
                                                         'rua_principal_sim': [1],
                                                         'quarto_hospedes_sim': [0],
                                                         'porao_sim': [0],
                                                         'aquecimento_agua_sim': [0],
                                                         'ar_condicionado_sim': [1],
                                                         'vagas_garagem': [2],
                                                         'local_pref_sim': [0],
                                                         'mobilia_sem_mobilia': [0]}))

# Valor predito pelo modelo BC
print(f"Valor Predito (Box-Cox): {round(valor_pred_bc[0], 2)}")

# Cálculo inverso para a obtenção do valor predito Y (preço)
valor_pred_preco = (valor_pred_bc * lmbda + 1) ** (1 / lmbda)
print(f"Valor Predito (Preço): {round(valor_pred_preco[0], 2)}")

#%% Gráfico fitted values

# Valores preditos pelo modelo para as observações da amostra
dados['fitted_bc'] = modelo_stepwise_bc.predict()

sns.regplot(dados, x='preco_bc', y='fitted_bc', color='blue', ci=False, line_kws={'color': 'red'})
plt.title('Analisando o Ajuste das Previsões', fontsize=10)
plt.xlabel('Preço Observado (Box-Cox)', fontsize=10)
plt.ylabel('Preço Previsto pelo Modelo (Box-Cox)', fontsize=10)
plt.axline((5.95, 5.95), (max(dados['preco_bc']), max(dados['preco_bc'])), linewidth=1, color='grey')
plt.show()

#%% Criação da função para o teste de Breusch-Pagan (heterocedasticidade)

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value

#%% Aplicando a função criada para realizar o teste

teste_bp = breusch_pagan_test(modelo_stepwise_bc)

# Tomando a decisão por meio do teste de hipóteses

alpha = 0.05 # nível de significância do teste

if teste_bp[1] > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

#%% Analisando a presença de heterocedasticidade no modelo original

teste_bp_original = breusch_pagan_test(reg)

# Tomando a decisão por meio do teste de hipóteses

alpha = 0.05 # nível de significância do teste

if teste_bp_original[1] > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

## O modelo com a transformação de Box-Cox ajustou os termos
## de erros heterocedásticos, indicando potencial erro 
## da forma funcional do modelo originalmente estimado

#%% Fim!