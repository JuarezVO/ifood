import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

import xgboost as xgb

from time import perf_counter

ts = perf_counter()

# Carrega o dataset e seleciona as features
df = pd.read_csv("dataset/full_dataset_clustered.csv")
print('Tamanho dos clusters: \n', df['cluster'].value_counts(),'\n')
features = ['channels', 'offer_type', 'min_value', 'discount_value', 'discount_per_minvalue', 'offer_success','cluster','duration','age','credit_card_limit']
df = df[features]
c1 = df[df['cluster'] == 1].drop(columns=['cluster'])

# One-hot encoding: cria colunas binárias para cada categoria de cada feature categórica
categorical_cols = [col for col in c1.columns if c1[col].dtype == 'object' and col != 'offer_success']
c1_encoded = pd.get_dummies(c1, columns=categorical_cols, dtype=int)

# Separa as features e alvo
x = c1_encoded.drop(columns=['offer_success'])
y = c1_encoded['offer_success']

# Separa os dados de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
train_data = pd.concat([x_train, y_train], axis=1)
print("Dataset:\n", train_data['offer_success'].value_counts(),'\n')

# Ajusta o desbalanceamento das classes
smote = SMOTE(random_state=42,)
x_train, y_train = smote.fit_resample(x_train, y_train)
train_balanced = pd.concat([x_train, y_train], axis=1)
print("Dataset balanceado:\n", train_balanced['offer_success'].value_counts(),'\n')

# Define os modelos
tree = DecisionTreeClassifier(random_state=42)
xgboost = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Define os parâmetros para o GridSearchCV
params = {
    'max_depth': [7, 9, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [100, 150],
    'max_features': ['sqrt', 'log2', None]
}
params_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Ajusta o modelo
grid = GridSearchCV(tree, params, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)

# Apresenta os resultados
best_tree = grid.best_estimator_
print(grid.best_params_)
for feature, importance in sorted(zip(x.columns, grid.best_estimator_.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance}")

report = classification_report(y_test, best_tree.predict(x_test), output_dict=True)
print(pd.DataFrame(report))

# Taxa de conversão atual (sem modelo)
baseline = c1['offer_success'].mean()
modelo = report['1']['precision']
ganho = (modelo - baseline) / baseline * 100

print(f'Base: {baseline*100.0:.2f}% | Modelo: {modelo*100.0:.2f}% | Melhoria estimada: {ganho:.2f}%')
print(f'Time: {round((perf_counter() - ts)/60, 2)}min')


# feature_names = []
# for col in x.columns:
#     if col in encoders:
#         feature_names.append(f"{col}: {list(encoders[col].classes_)}")
#     else:
#         feature_names.append(col)

# plt.figure(figsize=(20, 8))
# plot_tree(best_tree,
#           feature_names=feature_names,
#           class_names=['Failed', 'Success'],
#           filled=True,
#           rounded=True,
#           fontsize=8)
# plt.show()
