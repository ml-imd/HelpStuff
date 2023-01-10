# Algortimo para fazer vários testes de diferentes modelos de regressão
A última coluna dos datasets passados irão ser a variável de alvo y
### Parâmetros
```bash
-h, --help
```
Mostra ajuda.
```bash
-f, --file
```
Lista com caminho para os arquivos ou para diretórios onde estão os datasets. Ex: `-f .\datasets\` para listar todos um diretório `-f "[.\
datasets\iris.data,.\datasets\cancer.data]"` para arquivos em diferentes pastes
Mostra ajuda.
```bash
-s, --sep
```
Lista com métodos de separação, caso deseja utilizar Holdout basta colocar o número representando a proporção de dados de teste, caso deseje utilziar  Kfold basta colocar um número maior que um. Ex `-s ""[3,0.3,0.15]"` aqui será realizado um Kfold de 3, um holdtou com 30% de dados de treino e outro com 15% de dados de treino.

```bash
-me, --metrics
```
Métricas utilizadas para avaliar o modelo, erros podem acontecer pois algumas são apenas para classificação binária. Ex: `-me "[acc,f1,confusion_matrix]"`
Lista de métricas disponíveis:
- accuracy
- acc
- balanced_accuracy
- top_k_accuracy
- average_precision
- neg_brier_score
- f1
- f1_micro
- f1_macro
- f1_weighted
- f1_samples
- neg_log_loss
- precision
- recall
- jaccard
- roc_auc
- roc_auc_ovr
- roc_auc_ovo
- roc_auc_ovr_weighted
- roc_auc_ovo_weighted
- matrix
- confusion_matrix

```
-mo, --model
```
Lista com modelos utilizados. Ex: `-mo "
[rf,knn,AdaBoostClassifier]"`. Modelos disponíveis:
- knn ou KNeighborsClassifier
- RandomForestClassifier ou rf
- MLPClassifier ou mlp
- AdaBoostClassifier
- BernoulliNB
- CategoricalNB
- ComplementNB
- DummyClassifier
- ExtraTreesClassifier
- GaussianProcessClassifier
- HistGradientBoostingClassifier
- LabelPropagation
- LinearDiscriminantAnalysis
- LogisticRegression
- MultinomialNB
- NuSVC
- OneVsRestClassifier
- PassiveAggressiveClassifier
- QuadraticDiscriminantAnalysis
- RidgeClassifierCV
- SVC
- VotingClassifier

```
-n, --name
```
Nome do arquivo gerado.

### Exemplo de aplicação
```bash
python .\classification_exp.py -f "[.\datasets\iris.data,.\datasets\cancer.data]"  -s "[3,0.3,0.15]" -me "[acc]" -mo "[rf,knn,AdaBoostClassifier]" -n experimentos_finais
```