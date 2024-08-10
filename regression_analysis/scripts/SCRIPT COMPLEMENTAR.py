modelo.params
modelo.rsquared

#%%
# Somatória dos quadrados do modelo (SQM)
modelo.ess

# Somatória dos erros ao quadrado (SQErro)
modelo.ssr

#%%
n = modelo.nobs
n

df_modelo = modelo.df_model
df_modelo

df_residuos = modelo.df_resid
df_residuos

#%%
# Cálculo da estatística F de Fisher/Snedecor
F = ((modelo.ess)/df_modelo)/((modelo.ssr)/df_residuos)
F

#%%
# ANOVA (Analysis of Variance)
from statsmodels.stats.anova import anova_lm
anova_lm(modelo)

anova_lm(modelo).F.iloc[0]

#%%
# Definição do p-value associado ao F calculado
from scipy.stats import f

1 - f.cdf(F, df_modelo, df_residuos)

# Portanto, HÁ MODELO!

#%%
# F Hipotético
df1 = 10
df2 = 200

f_values = np.random.f(df1, df2, 100000)

plt.hist(f_values, bins=100, edgecolor='black')
plt.hist(f_values, bins=100, color='lightgrey')

#%%
# Voltando ao nosso modelo
# Cálculo do F crítico

f.ppf(0.95, df_modelo, df_residuos)

#%%
# Distribuição t de Student
# Hipotética

df = 40
t_values = np.random.standard_t(df, 100000)

plt.hist(t_values, bins=100, edgecolor='black')
plt.hist(t_values, bins=100, color='lightgrey')

#%%
# Cálculo da estatística t do beta da variável 'distancia' para o nosso exemplo
# t é a raiz quadrada de F (regressão simples)

t_estat = np.sqrt(F)
t_estat

# Cálculo do p-value da estatística t
from scipy.stats import t

t.sf(t_estat, df_residuos)*2

#%%
# Gráfico de distribuições t hipotéticas

x = np.arange(-4,4,0.0001)

plt.plot(x,t.pdf(x,2), label='df=2', color='yellow')
plt.plot(x,t.pdf(x,8), label='df=8', color='green')
plt.plot(x,t.pdf(x,20), label='df=20', color='darkorchid')
plt.legend()

#%% Importção para a geração da figura com imagem
from PIL import Image

#%%
# EXEMPLO 3:
df_corrupcao.groupby('regiao')['cpi'].mean().reset_index()

#%%
# Mudando a categoria de referência

df_corrupcao_dummies = pd.get_dummies(df_corrupcao, columns=['regiao'],
                                      dtype=int,
                                      drop_first=False) # n dummies

df_corrupcao_dummies

#%%
# Categoria de referência = Oceania

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_corrupcao_dummies.drop(columns=['cpi','pais',
                                                        'regiao_numerico',
                                                        'regiao_Oceania']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "cpi ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

# Estimação
modelo_corrupcao_dummies = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_corrupcao_dummies).fit()

# Parâmetros do 'modelo_corrupcao_dummies'
modelo_corrupcao_dummies.summary()