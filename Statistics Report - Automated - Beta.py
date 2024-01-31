#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importando bibliotecas

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mtbpy as mtb
from reportlab.pdfgen import canvas
from prettytable import PrettyTable
import statsmodels.api as sm
from scipy.stats import shapiro
from outliers import smirnov_grubbs as grubbs
import scipy.stats as stats


# In[2]:


df = pd.read_csv('Teste_Linearidade.csv')
pdf = canvas.Canvas('C:/Users/guilh/Área de Trabalho/Relatório_Linearidade')


# In[3]:


def format_for_print(df):    
    table = PrettyTable(list(df.columns))
    for row in df.itertuples():
        table.add_row(row[1:])
    return str(table)


# In[4]:


# Coleta de dados
print(format_for_print(df))


# # Fazendo regressão linear - Métodos dos mínimos quadrados ordinários

# In[5]:


y = df[df.columns[2]]
x = df[df.columns[1]]

x = sm.add_constant(x)

model = sm.OLS(y,x).fit()

print(model.summary())


# # Teste do coeficiente angular

# In[6]:


p_coef_test_ang = float(model.pvalues[1])
p_coef_test_lin = float(model.pvalues[0])

if p_coef_test_ang <= 0.05:
    print("Como P-valor ({:.3f}) do teste F da ANOVA é menor que 0,05 (conforme especificado) rejeitamos a hipótese nula (coeficiente angular igual zero) ao nível de significância de 5%."
          .format(p_coef_test_ang))
else:
    print("Como P-valor ({:.3f}) do teste F da ANOVA é maior que 0,05 (conforme especificado) não rejeitamos a hipótese nula (coeficiente angular igual zero) ao nível de significância de 5%."
          .format(p_coef_test_ang))


# # Teste do intercepto

# In[7]:


if p_coef_test_lin <= 0.05:
    print("Como P-valor ({:.3f}) do teste t é menor que 0,05, rejeitamos a hipótese nula (intercepto igual a zero) ao nível de significância de 5%."
          .format(p_coef_test_lin))
else:
    print("Como P-valor ({:.3f}) do teste t é maior que 0,05, não rejeitamos a hipótese nula (intercepto igual a zero) ao nível de significância de 5%."
          .format(p_coef_test_lin))


# ### Tabela - Impacto do coeficiente

# In[8]:


impacto_coef_linear = df.copy(deep= True)
impacto_coef_linear['Impacto do Coeficiente Linear (%)'] = round(model.params[0] / impacto_coef_linear[impacto_coef_linear.columns[2]] * 100,
                                                                 ndigits = 5)
print(format_for_print(impacto_coef_linear))


# # Teste de Correlação do modelo

# ### Coeficiente de Correlação de Pearson

# In[9]:


tabela_correlacao = pd.DataFrame(data = {'Graus de liberdade': [model.df_resid],
                                         'Coef. Determ.': [round(model.rsquared, ndigits = 4)],
                                         'Coef. Correlação': [round(model.rsquared ** 0.5, ndigits = 4)]})

print(format_for_print(tabela_correlacao))


# # Análise Gráfica

# ### Gráfico de Regressão

# In[10]:


plt.figure(figsize= (12,8))
sns.set_theme(style= 'whitegrid', font= 'Arial')

plt.title('Gráfico de Regressão', fontsize= 20)
grafico_disp = sns.regplot(x = df.columns[1],
                           y = df.columns[2],
                           data= df)
plt.show()


# ### Diagnóstico dos Resíduos do Modelo

# In[11]:


influence = model.get_influence()
std_resid = influence.resid_studentized_internal
student_resid = influence.resid_studentized_external
cooks_d = influence.cooks_distance


fig, ax = plt.subplots(2, 2, sharex='none', sharey='none',
                       figsize= (14,10))

ax[0,0].plot(model.fittedvalues,std_resid, 'o', color= 'gray')
ax[0,0].hlines(y= [-2,2], xmin= model.fittedvalues.min()-1, xmax= model.fittedvalues.max()+1, color= 'blue', linestyle= '--')
ax[0,0].hlines(y= [-3,3], xmin= model.fittedvalues.min()-1, xmax= model.fittedvalues.max()+1, color= 'green', linestyle= '--')
ax[0,0].set(title= 'Res. Padronizados vs Valores Ajustados',
            xlabel= 'Valor Ajustado',
            ylabel= 'Resíduos Padronizados')

ax[1,0].plot(model.fittedvalues,model.resid, 'o', color= 'gray')
ax[1,0].hlines(y= 0, xmin= model.fittedvalues.min()-1, xmax= model.fittedvalues.max()+1, color= 'blue', linestyle= '--')
ax[1,0].set(title= 'Resíduos vs Valores Ajustados',
            xlabel= 'Valor Ajustado',
            ylabel= 'Resíduos')

sm.qqplot(model.resid, line='s', ax = ax[0,1], fit= True, )
ax[0,1].set(title= 'QQ-Plot',
            xlabel= 'Quantis da Normal',
            ylabel= 'Resíduos')

ax[1,1].plot(model.resid, 'o--', color='gray')
ax[1,1].hlines(y= 0, xmin= -1, xmax= model.resid.count(), color= 'blue', linestyle= '--')
ax[1,1].set(title= 'Resíduos vs Ordem de Coleta',
            xlabel= 'Ordem de Coleta',
            ylabel= 'Resíduos')

plt.show()


# # Teste de normalidade dos resíduos

# ### Teste de Shapiro-Wilk

# In[12]:


tabela_normalidade = pd.DataFrame(data = {' ': 'Shapiro-Wilk',
                                         'Estatística': [round(shapiro(model.resid)[0], ndigits= 6)],
                                         'P-Valor': [round(shapiro(model.resid)[1], ndigits= 5)]})

print(format_for_print(tabela_normalidade))

if shapiro(model.resid)[1] <= 0.05:
    print("Como P-valor ({:.3f}) do teste de Shapiro-Wilk é menor que 0,05, rejeitamos a hipótese de normalidade dos resíduos ao nível de significância de 5%."
          .format(shapiro(model.resid)[1]))
else:
    print("Como P-valor ({:.3f}) do teste de Shapiro-Wilk é maior que 0,05, aceitamos a hipótese de normalidade dos resíduos ao nível de significância de 5%."
          .format(shapiro(model.resid)[1]))


# # Teste de Heterocedasticidade

# In[13]:


teste_bp = sm.stats.het_breuschpagan(model.resid, model.model.exog)

tabela_bp = pd.DataFrame(data = {' ': 'Breusch-Pagan',
                                 'Estatística': [round(teste_bp[0], ndigits= 6)],
                                 'GL': [round(model.df_model, ndigits= 0)],
                                 'P-Valor': [round(teste_bp[1], ndigits= 5)]})

print(format_for_print(tabela_bp))

if teste_bp[1] < 0.05:
    print("Como P-valor ({:.3f}) do Teste de Breusch Pagan é menor que 0,05 (conforme proposto), rejeitamos a hipótese de igualdade das variâncias ao nível de significância de 5%. Logo, temos um modelo heterocedástico."
          .format(teste_bp[1]))
else:
    print("Como P-valor ({:.3f}) do Teste de Breusch Pagan é maior que 0,05 (conforme proposto), não rejeitamos a hipótese de igualdade das variâncias ao nível de significância de 5%. Logo, temos um modelo homocedástico."
          .format(teste_bp[1]))


# # Caso o modelo seja heterocedástico - Regressão pelo método dos mínimos quadrados ponderados

# ## Fazendo a regressão - Método dos mínimos quadrados ponderados

# In[37]:


pontos = df[df.columns[0]].unique()
pontos.sort()
var_por_ponto = df[df.columns[2]].groupby(df[df.columns[0]]).var()
cod_w = pd.DataFrame({df.columns[0]:pontos,
                      'Pesos': len(var_por_ponto) * sum(var_por_ponto)/var_por_ponto})

weights = []
for row in df[df.columns[0]]:
    weights.append(round(float(cod_w['Pesos'].loc[cod_w[cod_w.columns[0]] == row]), ndigits= 6))

if teste_bp[1] < 0.05:
    model = sm.WLS(y, x, weights= weights).fit()
    
    print(model.summary())   


# ## Teste do Coeficiente Angular

# In[15]:


if teste_bp[1] < 0.05:
    p_coef_test_ang = float(model.pvalues[1])
    p_coef_test_lin = float(model.pvalues[0])

    if p_coef_test_ang <= 0.05:
        print("Como P-valor ({:.3f}) do teste F da ANOVA é menor que 0,05 (conforme especificado) rejeitamos a hipótese nula (coeficiente angular igual zero) ao nível de significância de 5%."
              .format(p_coef_test_ang))
    else:
        print("Como P-valor ({:.3f}) do teste F da ANOVA é maior que 0,05 (conforme especificado) não rejeitamos a hipótese nula (coeficiente angular igual zero) ao nível de significância de 5%."
              .format(p_coef_test_ang))


# ## Teste do Intercepto

# In[16]:


if teste_bp[1] < 0.05:
    if p_coef_test_lin <= 0.05:
        print("Como P-valor ({:.3f}) do teste t é menor que 0,05, rejeitamos a hipótese nula (intercepto igual a zero) ao nível de significância de 5%."
              .format(p_coef_test_lin))
    else:
        print("Como P-valor ({:.3f}) do teste t é maior que 0,05, não rejeitamos a hipótese nula (intercepto igual a zero) ao nível de significância de 5%."
              .format(p_coef_test_lin))


# ### Tabela de Impacto do Coeficiente

# In[17]:


if teste_bp[1] < 0.05:
    impacto_coef_linear = df.copy(deep= True)
    impacto_coef_linear['Impacto do Coeficiente Linear (%)'] = round(model.params[0] / impacto_coef_linear[impacto_coef_linear.columns[2]] * 100,
                                                                     ndigits = 5)
    print(format_for_print(impacto_coef_linear))


# ## Teste de Correlação do Modelo

# ### Coeficiente de Correlação de Pearson

# In[18]:


if teste_bp[1] < 0.05:
    tabela_correlacao = pd.DataFrame(data = {'Graus de liberdade': [model.df_resid],
                                             'Coef. Determ.': [round(model.rsquared, ndigits = 4)],
                                             'Coef. Correlação': [round(model.rsquared ** 0.5, ndigits = 4)]})

    print(format_for_print(tabela_correlacao))


# ## Análise Gráfica

# ### Gráfico de Regressão

# In[19]:


if teste_bp[1] < 0.05:
    plt.figure(figsize= (12,8))
    sns.set_theme(style= 'whitegrid', font= 'Arial')

    plt.title('Gráfico de Regressão', fontsize= 20)
    grafico_disp = sns.regplot(x = df.columns[1],
                               y = df.columns[2],
                               data= df)
    plt.show()


# ### Diagnóstico de Resíduos do Modelo

# In[44]:


if teste_bp[1] > 0.05:
    #influence = model.get_influence()
    #std_resid = influence.resid_studentized_internal
    #student_resid = influence.resid_studentized_external
    #cooks_d = influence.cooks_distance
    
    residuals = model.resid
    mse_resid = np.mean(residuals**2)
    covariance_matrix = np.linalg.inv(np.dot(x.T * weights, x))
    hat_matrix = np.dot(x, np.dot(covariance_matrix, x.T))

    std_resid = residuals / np.sqrt(mse_resid * (1 - np.diag(hat_matrix)))
    student_resid = residuals / np.sqrt(mse_resid * (1 - np.diag(hat_matrix))) / np.sqrt(1 - np.diag(hat_matrix))
    cooks_d = residuals**2 / mse_resid / np.diag(hat_matrix) / (x.shape[0] - x.shape[1])

    fig, ax = plt.subplots(2, 2, sharex='none', sharey='none',
                           figsize= (14,10))

    ax[0,0].plot(model.fittedvalues,std_resid, 'o', color= 'gray')
    ax[0,0].hlines(y= [-2,2], xmin= model.fittedvalues.min()-1, xmax= model.fittedvalues.max()+1, color= 'blue', linestyle= '--')
    ax[0,0].hlines(y= [-3,3], xmin= model.fittedvalues.min()-1, xmax= model.fittedvalues.max()+1, color= 'green', linestyle= '--')
    ax[0,0].set(title= 'Res. Padronizados vs Valores Ajustados',
                xlabel= 'Valor Ajustado',
                ylabel= 'Resíduos Padronizados')

    ax[1,0].plot(model.fittedvalues,model.resid, 'o', color= 'gray')
    ax[1,0].hlines(y= 0, xmin= model.fittedvalues.min()-1, xmax= model.fittedvalues.max()+1, color= 'blue', linestyle= '--')
    ax[1,0].set(title= 'Resíduos vs Valores Ajustados',
                xlabel= 'Valor Ajustado',
                ylabel= 'Resíduos')

    sm.qqplot(model.resid, line='s', ax = ax[0,1], fit= True, )
    ax[0,1].set(title= 'QQ-Plot',
                xlabel= 'Quantis da Normal',
                ylabel= 'Resíduos')

    ax[1,1].plot(model.resid, 'o--', color='gray')
    ax[1,1].hlines(y= 0, xmin= -1, xmax= model.resid.count(), color= 'blue', linestyle= '--')
    ax[1,1].set(title= 'Resíduos vs Ordem de Coleta',
                xlabel= 'Ordem de Coleta',
                ylabel= 'Resíduos')

    plt.show()


# ## Teste de Normalidade dos Resíduos

# ### Teste de Shapiro-Wilk

# In[55]:


if teste_bp[1] > 0.05:
    tabela_normalidade = pd.DataFrame(data = {' ': 'Shapiro-Wilk',
                                             'Estatística': [round(shapiro(model.wresid)[0], ndigits= 6)],
                                             'P-Valor': [round(shapiro(model.wresid)[1], ndigits= 5)]})

    print(format_for_print(tabela_normalidade))

    if shapiro(model.wresid)[1] <= 0.05:
        print("Como P-valor ({:.3f}) do teste de Shapiro-Wilk é menor que 0,05, rejeitamos a hipótese de normalidade dos resíduos ao nível de significância de 5%."
              .format(shapiro(model.wresid)[1]))
    else:
        print("Como P-valor ({:.3f}) do teste de Shapiro-Wilk é maior que 0,05, aceitamos a hipótese de normalidade dos resíduos ao nível de significância de 5%."
              .format(shapiro(model.wresid)[1]))


# ## Teste de Heterocedasticidade

# In[52]:


if teste_bp[1] < 0.05:
    teste_bp_pond = sm.stats.het_breuschpagan(model.wresid, model.model.exog)

    tabela_bp_pond = pd.DataFrame(data = {' ': 'Breusch-Pagan',
                                         'Estatística': [round(teste_bp_pond[0], ndigits= 6)],
                                         'GL': [round(model.df_model, ndigits= 0)],
                                         'P-Valor': [round(teste_bp_pond[1], ndigits= 5)]})

    print(format_for_print(tabela_bp_pond))

    if teste_bp_pond[1] < 0.05:
        print("Como P-valor ({:.3f}) do Teste de Breusch Pagan é menor que 0,05 (conforme proposto), rejeitamos a hipótese de igualdade das variâncias ao nível de significância de 5%. Logo, temos um modelo heterocedástico."
              .format(teste_bp_pond[1]))
    else:
        print("Como P-valor ({:.3f}) do Teste de Breusch Pagan é maior que 0,05 (conforme proposto), não rejeitamos a hipótese de igualdade das variâncias ao nível de significância de 5%. Logo, temos um modelo homocedástico."
              .format(teste_bp_pond[1]))


# # Avaliação de Outliers

# ### Tabela com Valores resumidos

# In[48]:


tabela_std_resid = df.copy(deep= True)
tabela_std_resid['Resíduos Padronizados'] = std_resid.round(6)

print(format_for_print(tabela_std_resid))


# ### Teste de Grubbs

# In[49]:


n_std_resid = len(std_resid)
mean_std_resid = np.mean(std_resid)
sd_std_resid = np.std(std_resid)
numerator = max(abs(std_resid - mean_std_resid))
g_calculated = numerator / sd_std_resid

t_value_1 = stats.t.ppf(1 - 0.05 / (2 * n_std_resid), n_std_resid - 2)
g_critical = ((n_std_resid - 1) * np.sqrt(np.square(t_value_1))) / (np.sqrt(n_std_resid) * np.sqrt(n_std_resid - 2 + np.square(t_value_1)))

tabela_grubbs = pd.DataFrame(data = {' ': 'Grubbs',
                             'Estat. G': [round(g_calculated, ndigits= 6)],
                             'G crítico': [round(g_critical, ndigits= 6)]})

print(format_for_print(tabela_grubbs))

if g_critical > g_calculated:
  print("Como estatística G ({:.5f}) é menor que G crítico ({:.5f}) para o Teste de Grubbs, aceitamos a hipotese nula (todos os dados os dados pertencem a mesma população)."
        .format(g_calculated,g_critical))
else:
  print("Como estatística G ({:.5f}) é maior que G crítico ({:.5f}) para o Teste de Grubbs, rejeitamos a hipotese nula (um ou mais valores são outliers)."
        .format(g_calculated,g_critical))


# # Teste de Independência das Observações

# ### Criação da Tabela com Valores Críticos para Durbin-Watson

# In[25]:


dic_dw = {'Tamanho amostral': [15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60],
          'DL': [1.07697, 1.15759, 1.22115, 1.27276, 1.31568, 1.35204, 1.38335, 1.41065, 1.43473, 1.45615, 1.47538, 1.49275, 1.50856, 1.523, 1.53628, 1.54853],
          'DU': [1.36054, 1.39133, 1.41997, 1.44575, 1.46878, 1.48936, 1.50784, 1.52451, 1.53963, 1.5534, 1.56602, 1.57762, 1.58835, 1.59829, 1.60754, 1.61617]}

tabela_dw = pd.DataFrame(dic_dw)


# ### Teste de independência dos resíduos - Durbin-Watson

# In[51]:


if teste_bp[1] > 0.05:
    result_dw = round(sm.stats.stattools.durbin_watson(model.resid), ndigits= 5)
else
    result_dw = round(sm.stats.stattools.durbin_watson(model.wresid), ndigits= 5)

tabela_result_dw = pd.DataFrame(data = {' ': 'Durbin-Watson',
                             'Estat. D': [result_dw],
                             'Estat. D-4': [4 - result_dw]})
tabela_result_dw[['DL','DU']] = tabela_dw[['DL','DU']].loc[tabela_dw['Tamanho amostral'] == len(model.resid)]

print(format_for_print(tabela_result_dw))

estat_d = float(tabela_result_dw['Estat. D'])
estat_d_4 = float(tabela_result_dw['Estat. D-4'])
estat_DL = float(tabela_result_dw['DL'])
estat_DU = float(tabela_result_dw['DU'])

if estat_d > estat_DU:
    print('Como a estatística D ({:.5f}) é maior que DU ({:.5f}) é possível concluir que não existe autocorrelação positiva entre resíduos.'
          .format(estat_d,estat_DU))
elif estat_d < estat_DU:
    if estat_d > estat_DL:
        print('Como a estatística D ({:.5f}) é menor que DU ({:.5f}) e maior que DL ({:.5f}) o teste é inconclusivo para autocorrelação positiva entre resíduos.'
              .format(estat_d,estat_DU,estat_DL))
    elif estat_d < estat_DL:
        print('Como a estatística D ({:.5f}) é menor que DU ({:.5f}) e menor que DL ({:.5f}) é possível concluir que existe autocorrelação positiva entre resíduos.'
              .format(estat_d,estat_DU,estat_DL))
        
if estat_d_4 > estat_DU:
    print('Como a estatística D ({:.5f}) é maior que DU ({:.5f}) é possível concluir que não existe autocorrelação negativa entre resíduos.'
          .format(estat_d_4,estat_DU))
elif estat_d_4 < estat_DU:
    if estat_d_4 > estat_DL:
        print('Como a estatística D ({:.5f}) é menor que DU ({:.5f}) e maior que DL ({:.5f}) o teste é inconclusivo para autocorrelação negativa entre resíduos.'
              .format(estat_d_4,estat_DU,estat_DL))
    elif estat_d_4 < estat_DL:
        print('Como a estatística D ({:.5f}) é menor que DU ({:.5f}) e menor que DL ({:.5f}) é possível concluir que existe autocorrelação negativa entre resíduos.'
              .format(estat_d_4,estat_DU,estat_DL))
    


# In[ ]:




