import pandas as pd
from sklearn.datasets import make_classification
from sklearn.utils import all_estimators
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from sklearn import metrics
from itertools import product
from sklearn.model_selection import KFold
import json
import sys, getopt
import os
import ast
import time
from datetime import timedelta

resultados = {"dataset":[],
                  "metodo":[],
                  "modelo":[],
                  "tempo_execucao":[],
                 "score":[]}
count = 0 
caminho=''
RANDOM= 42
name='results'
metricas={
      "accuracy": metrics.accuracy_score
      ,"acc": metrics.accuracy_score
      ,"balanced_accuracy": metrics.balanced_accuracy_score
      ,"top_k_accuracy": metrics.top_k_accuracy_score
      ,"average_precision": metrics.average_precision_score
      ,"neg_brier_score": metrics.brier_score_loss
      ,"f1": metrics.f1_score
      ,"f1_micro": metrics.f1_score
      ,"f1_macro": metrics.f1_score
      ,"f1_weighted": metrics.f1_score
      ,"f1_samples": metrics.f1_score
      ,"neg_log_loss": metrics.log_loss
      ,"precision": metrics.precision_score
      ,"recall": metrics.recall_score
      ,"jaccard": metrics.jaccard_score
      ,"roc_auc": metrics.roc_auc_score
      ,"roc_auc_ovr": metrics.roc_auc_score
      ,"roc_auc_ovo": metrics.roc_auc_score
      ,"roc_auc_ovr_weighted": metrics.roc_auc_score
      ,"roc_auc_ovo_weighted": metrics.roc_auc_score
      ,"matrix": metrics.confusion_matrix
      ,"confusion_matrix": metrics.confusion_matrix
           }

estimators =all_estimators(type_filter='classifier')
estimators = {item[0]: item[1] for item in estimators}
estimators['dt'] = estimators['DecisionTreeClassifier']
estimators['knn'] = estimators['KNeighborsClassifier']
estimators['mlp'] = estimators['MLPClassifier']
estimators['rf'] = estimators['RandomForestClassifier']
estimators['nb'] = estimators['GaussianNB']

def ajuda():
  var = []
  lista_est = list(estimators.keys())
  for k in lista_est:
    var.append(k)
    lista_est.remove(k)
    for k1 in lista_est:
      if estimators[k] == estimators[k1]:
        var[-1]+=f' ou {k1}'
        var = var[-1:] + var[:-1]
        lista_est.remove(k1)
  print ('classification_exp.py <arquivo> <modelos>')
  print('Para seleção dos métodos de separação caso o número seja inteiro ele será um Kfold, caso contrário holdout')
  print('Métricas disponíveis:', *metricas.keys(), sep='\n- ')
  print('Modelos disponíveis:',*var, sep='\n- ')
  return

def converter_paths(paths):
  if paths[0] == '[' and paths[-1] == ']':
    return paths.strip('][').split(',')
  if os.path.isdir(paths):
    res = []
    for file_path in os.listdir(paths):
      if os.path.isfile(os.path.join(paths, file_path)):
          if paths[-1] != '"\\"':
            paths+='\\'
          res.append(paths+file_path)
    return res
  else:
    return [paths]

def check_sep(num):
  if num > 1:
    return 'Kfold'
  else:
    return 'Holdout'
  
def converter_sep(comando):
  lista = comando.strip('][').split(',')
  lista = [float(i) for i in lista]
  return lista
#Aqui irie converter as strings que foram entradas no argumento do modelo
#para seu formato verdadeiro
def converter_args(args):
  for key in args:
    if args[key].isdigit():
      args[key] = float(args[key])
      if args[key].is_integer():
        args[key] = int(args[key])
  return args
        

def treinar(X_train, X_test, y_train, y_test):
  global resultados
  global input_modelos
  global input_metricas
  for k in input_modelos:
    start = time.time()
    if len(k[1]) > 1:
      #Dividir a string dos parâmetros em dicionário
      args = dict(e.split('=') for e in k[1].split(','))
      args=converter_args(args)
      modelo = estimators[k[0]](**args)
    else:
      modelo = estimators[k[0]]()
    
    print(modelo)
    y_pred = modelo.fit(X_train,y_train).predict(X_test)
    resultado_metrica = {}
    for avaliar in input_metricas:
        resultado_metrica[avaliar] = metricas[avaliar](y_test,y_pred)
         
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    resultados['tempo_execucao'].append(elapsed) 
    resultados['modelo'].append(modelo) 
    resultados['score'].append(resultado_metrica) 
  return
if __name__ == "__main__":
  for i in range(1,len(sys.argv)):
    if sys.argv[i] == '-h' or sys.argv[i] == '--help':
      ajuda()
      break
    if sys.argv[i] == '--file' or sys.argv[i] == '-f':
      input_paths = converter_paths(sys.argv[i+1])
      print(f'{len(input_paths)} datasets:\n', *input_paths, sep=' | ')
    if sys.argv[i] =='-s' or sys.argv[i] == '-sep':
      input_separacao = converter_sep(sys.argv[i+1])
      print(f'{len(input_separacao)} métodos de separações\n', *[f'{check_sep(i)} {i}' for i in input_separacao], sep='| ')
    if sys.argv[i] =='-me' or sys.argv[i] == '--metrics':
      input_metricas = sys.argv[i+1].strip('][').split(',') 
      print(f'{len(input_metricas)} métricas:\n', *input_metricas, sep=' | ')
    if sys.argv[i] =='-mo' or sys.argv[i] == '--model':
      input_modelos = sys.argv[i+1][1:-1].split('),')
      print(f'{len(input_modelos)} modelos: \n',*input_modelos, sep=' | ')
      input_modelos = [a.split('(', 1) for a in input_modelos]
    if sys.argv[i] =='-n' or sys.argv[i] == '--name':
      name = sys.argv[i+1]  
      
  for dataset in input_paths:
    print(f'Treinando {dataset}...')
    df = pd.read_csv(dataset)
    X = df.iloc[:,:-1]._get_numeric_data()
    y = df.iloc[:,-1]
    
    for sep in input_separacao:
      if sep < 1:
          X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=sep,
                                                                random_state=RANDOM)
          treinar(X_train, X_test, y_train, y_test)
          resultados['metodo'].extend([f'Holdout {sep}']*len(input_modelos))
          resultados['dataset'].extend([dataset]*len(input_modelos))
      else:
          sep = int(sep)
          kf = KFold(n_splits=sep, shuffle=True, random_state=RANDOM)
          count=1
          for train_index, test_index in kf.split(X):
              X_train, X_test = X.loc[train_index], X.loc[test_index]
              y_train, y_test = y.loc[train_index], y.loc[test_index]
              treinar(X_train, X_test, y_train, y_test)
              resultados['metodo'].extend([f'Kfold {count}']*len(input_modelos))
              count+=1
              resultados['dataset'].extend([dataset]*len(input_modelos))
      final = pd.DataFrame(resultados)
      final = pd.concat([final,final['score'].apply(pd.Series)],axis=1).drop(['score'],axis=1)
      final.to_pickle(f'./{name}.pkl') 