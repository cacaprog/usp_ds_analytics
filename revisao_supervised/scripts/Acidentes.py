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
!pip install scipy

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP,ZeroInflatedPoisson
import warnings
warnings.filterwarnings('ignore')

#%% Importando o banco de dados

acidentes = pd.read_excel('acidentes.xlsx')
## Fonte: Fávero e Belfiore (2024) Manual de Análise de Dados, Capítulo 14

acidentes.info()

#%% Estatísticas descritivas

# Variáveis métricas
acidentes[['acidentes', 'pop', 'idade']].describe()

# Variável categórica
acidentes['leiseca'].value_counts()

# Tabela de frequências de Y
acidentes['acidentes'].value_counts().sort_index()

#%% Histograma da variável dependente

plt.figure(figsize=(15,10))
ax = sns.barplot(data = acidentes['acidentes'].value_counts().sort_index(), color='purple')
ax.bar_label(ax.containers[0])
plt.xlabel('Acidentes por Semana', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

#%% Comparação média x variância da variável dependente

print(f"Média: {round(acidentes['acidentes'].mean(), 2)}")
print(f"Variância: {round(acidentes['acidentes'].var(), 2)}")

# Variância consideravelmente maior do que a média!

#%% Organização dos dados para os modelos "zero inflated"

## É necessária a separação entre os componentes (contagem e zero inflated)

# Variável dependente
y = acidentes['acidentes']

# Variáveis preditoras: componente de contagem
x1 = acidentes['pop']
X1 = sm.add_constant(x1)

# Variáveis preditoras: componente logit (zero inflated)
x2 = acidentes[['idade', 'leiseca']]
X2 = sm.add_constant(x2)

#%% Modelo Zero Inflated Poisson 

# Estimando o modelo
modelo_zip = sm.ZeroInflatedPoisson(y, X1, exog_infl=X2,
                                    inflation='logit').fit()

# Parâmetros do modelo
modelo_zip.summary()

# Valores preditos pelo modelo para observações da amostra
acidentes['poisson_zi'] = modelo_zip.predict(X1, exog_infl=X2)

#%% Modelo Poisson (comparação)

modelo_poisson = sm.Poisson.from_formula(formula='acidentes ~ pop', 
                                         data=acidentes).fit()

# Parâmetros do modelo
modelo_poisson.summary()

# Valores preditos pelo modelo para observações da amostra
acidentes['poisson'] = modelo_poisson.predict()

#%% Teste de Vuong (definição da função)

# VUONG, Q. H. Likelihood ratio tests for model selection and non-nested
#hypotheses. Econometrica, v. 57, n. 2, p. 307-333, 1989.

# Definição de função para elaboração do teste de Vuong
# Autores: Luiz Paulo Fávero e Helder Prado Santos

def vuong_test(m1, m2):

    from scipy.stats import norm    

    if m1.__class__.__name__ == "GLMResultsWrapper":
        
        glm_family = modelo_poisson.model.family

        X = pd.DataFrame(data=m1.model.exog, columns=m1.model.exog_names)
        y = pd.Series(m1.model.endog, name=m1.model.endog_names)

        if glm_family.__class__.__name__ == "Poisson":
            m1 = Poisson(endog=y, exog=X).fit()
            
        if glm_family.__class__.__name__ == "NegativeBinomial":
            m1 = NegativeBinomial(endog=y, exog=X, loglike_method='nb2').fit()

    supported_models = [ZeroInflatedPoisson,ZeroInflatedNegativeBinomialP,Poisson,NegativeBinomial]
    
    if type(m1.model) not in supported_models:
        raise ValueError(f"Model type not supported for first parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
        
    if type(m2.model) not in supported_models:
        raise ValueError(f"Model type not supported for second parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
    
    # Extração das variáveis dependentes dos modelos
    m1_y = m1.model.endog
    m2_y = m2.model.endog

    m1_n = len(m1_y)
    m2_n = len(m2_y)

    if m1_n == 0 or m2_n == 0:
        raise ValueError("Could not extract dependent variables from models.")

    if m1_n != m2_n:
        raise ValueError("Models appear to have different numbers of observations.\n"
                         f"Model 1 has {m1_n} observations.\n"
                         f"Model 2 has {m2_n} observations.")

    if np.any(m1_y != m2_y):
        raise ValueError("Models appear to have different values on dependent variables.")
        
    m1_linpred = pd.DataFrame(m1.predict(which="prob"))
    m2_linpred = pd.DataFrame(m2.predict(which="prob"))        

    m1_probs = np.repeat(np.nan, m1_n)
    m2_probs = np.repeat(np.nan, m2_n)

    which_col_m1 = [list(m1_linpred.columns).index(x) if x in list(m1_linpred.columns) else None for x in m1_y]    
    which_col_m2 = [list(m2_linpred.columns).index(x) if x in list(m2_linpred.columns) else None for x in m2_y]

    for i, v in enumerate(m1_probs):
        m1_probs[i] = m1_linpred.iloc[i, which_col_m1[i]]

    for i, v in enumerate(m2_probs):
        m2_probs[i] = m2_linpred.iloc[i, which_col_m2[i]]

    lm1p = np.log(m1_probs)
    lm2p = np.log(m2_probs)

    m = lm1p - lm2p

    v = np.sum(m) / (np.std(m) * np.sqrt(len(m)))

    pval = 1 - norm.cdf(v) if v > 0 else norm.cdf(v)

    print("Vuong Non-Nested Hypothesis Test-Statistic (Raw):")
    print(f"Vuong z-statistic: {round(v, 3)}")
    print(f"p-value: {pval:.3f}")
    print("")
    print("==================Result======================== \n")
    if pval <= 0.05:
        print("H1: Indicates inflation of zeros at 95% confidence level")
    else:
        print("H0: Indicates no inflation of zeros at 95% confidence level")

#%% Teste de Vuong (aplicação aos dados)

vuong_test(modelo_zip, modelo_poisson)

# O resultado teste evidencia a inflação de zeros

#%% Teste de razão de verossimilhança

# Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 3)
    print(f"χ²: {round(LR_statistic,2)}"), print(f"p-valor: {round(p_val,2)}")
    if p_val <= 0.05:
        print("H1: Modelos diferentes, favorecendo aquele com a maior Log-Likelihood")
    else:
        print("H0: Modelos com log-likelihoods que não são estatisticamente diferentes ao nível de confiança de 95%")

# Teste de razão de verossimilhança: Poisson GLM e ZI Poisson
lrtest([modelo_poisson, modelo_zip])

#%% Modelo ZI Binomial Negativo

# Estimando o modelo
modelo_zibn = ZeroInflatedNegativeBinomialP(y, X1, exog_infl=X2,
                                            inflation='logit').fit(method='nm', maxiter=1000)

# Parâmetros do modelo
modelo_zibn.summary()

# Interpretando o 'alpha'

# Se o p-valor alpha < nível de sig., é estatisticamente diferente de zero
# Portanto, observa-se superdispersão na variável dependente!

# Valores preditos pelo modelo para observações da amostra
acidentes['bn_zi'] = modelo_zibn.predict(X1, exog_infl=X2)

#%% Modelo Binomial Negativo (comparação)

modelo_bn = sm.NegativeBinomial.from_formula(formula='acidentes ~ pop', 
                                             data=acidentes).fit()

# Parâmetros do modelo
modelo_bn.summary()

# Interpretando o 'alpha'

# Se o p-valor alpha < nível de sig., é estatisticamente diferente de zero
# Portanto, observa-se superdispersão na variável dependente!

# Valores preditos pelo modelo para observações da amostra
acidentes['bn'] = modelo_bn.predict()

#%% Teste de razão de verossimilhança

# Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 3)
    print(f"χ²: {round(LR_statistic,2)}"), print(f"p-valor: {round(p_val,2)}")
    if p_val <= 0.05:
        print("H1: Modelos diferentes, favorecendo aquele com a maior Log-Likelihood")
    else:
        print("H0: Modelos com log-likelihoods que não são estatisticamente diferentes ao nível de confiança de 95%")

# Teste de razão de verossimilhança: Binomial Negativo GLM e ZIBN
lrtest([modelo_bn, modelo_zibn])

#%% Teste de razão de verossimilhança

# Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    print(f"χ²: {round(LR_statistic,2)}"), print(f"p-valor: {round(p_val,2)}")
    if p_val <= 0.05:
        print("H1: Modelos diferentes, favorecendo aquele com a maior Log-Likelihood")
    else:
        print("H0: Modelos com log-likelihoods que não são estatisticamente diferentes ao nível de confiança de 95%")

# Teste de razão de verossimilhança: ZIP e ZIBN
lrtest([modelo_zip, modelo_zibn])

#%% Visualizando graficamente as loglik

# Definição do dataframe com os modelos e respectivos LogLiks
df_llf = pd.DataFrame({'modelo':['ZIP','ZIBN', 'Poisson', 'BN'],
                       'loglik':[modelo_zip.llf, 
                                 modelo_zibn.llf, 
                                 modelo_poisson.llf, 
                                 modelo_bn.llf]}).sort_values(by=['loglik'])

# Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ['lightsalmon', 'darksalmon', 'coral', 'orangered']

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='snow', fontsize=25, weight='bold')
ax.set_ylabel("Modelo Proposto", fontsize=20)
ax.set_xlabel("LogLik", fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.show()

#%% Realizando predições no modelo ZIBN

pred_zibn = modelo_zibn.predict(pd.DataFrame({'const': [1],
                                              'pop': [1.5]}),
                                exog_infl=pd.DataFrame({'const': [1],
                                                        'idade': [38],
                                                        'leiseca': [1]}))

print(f"Quant. Estimada de Acidentes por Semana: {round(pred_zibn[0], 2)}")

#%% Comparando as previsões dos modelos

plt.figure(figsize=(15,10))
sns.regplot(data=acidentes, x=acidentes['pop'], y=acidentes['poisson'],
            ci=None, marker='o', lowess=True, scatter=False,
            label='Poisson',
            line_kws={'color':'orange', 'linewidth':5})
sns.regplot(data=acidentes, x=acidentes['pop'], y=acidentes['poisson_zi'],
            ci=None, marker='o', lowess=True, scatter=False,
            label='ZIP',
            line_kws={'color':'darkorchid', 'linewidth':5})
sns.regplot(data=acidentes, x=acidentes['pop'], y=acidentes['bn'],
            ci=None, marker='o', lowess=True, scatter=False,
            label='BNeg',
            line_kws={'color':'limegreen', 'linewidth':5})
sns.regplot(data=acidentes, x=acidentes['pop'], y=acidentes['bn_zi'],
            ci=None, marker='o', lowess=True, scatter=False,
            label='ZINB',
            line_kws={'color':'dodgerblue', 'linewidth':5})
sns.regplot(data=acidentes, x=acidentes['pop'], y=acidentes['acidentes'],
            ci=None, marker='o', fit_reg=False,
            scatter_kws={'color':'black', 's':120, 'alpha':0.5})
plt.xlabel('População urbana (x milhão)', fontsize=17)
plt.ylabel('Quantidade de acidentes de trânsito na última semana', fontsize=17)
plt.legend(fontsize=20)
plt.show()

#%% Fim!