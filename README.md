# Previsão de aceite de ofertas

Pipeline de machine learning em **pandas + scikit-learn** que agrupa usuários por perfil (K-Means) e prevê se uma oferta será aceita (árvore de decisão).

Infelizmente, como eu não uso linux, eu tive uma série de problemas com o PySpark, então eu adicionei ele como um branch separada:

```bash
git checkout spark
```

## Como utilizar

### Pré-requisitos

- Dados em `data/raw/` (`offers.json`, `profile.json`, `transactions.json`)

```bash
pip install -r requirements.txt
```

### Treinar os modelos

Na raiz do projeto:

```bash
python run_train.py
```

O script:

1. Prepara e une os três JSONs (`prep.py`, pandas)
2. Salva `data/processed/full_dataset.csv` (antes de excluir ofertas informacionais)
3. Gera gráficos exploratórios em `data/images/`
4. Agrupa usuários em 3 clusters, grava `data/processed/full_dataset_clustered.csv` e serializa modelos em `model/`
5. Treina a árvore de decisão no cluster alvo (`DECISION_TREE_TARGET_CLUSTER`, padrão **1**)

### Fazer previsões

Execute o treino antes. Depois edite o exemplo em `run_forecast.py` e rode:

```bash
python run_forecast.py
```

Fluxo:

1. Preencha `UserProfile` (idade, gênero, limite, valor médio, taxa de sucesso)
2. Preencha `UserOffer` (valores da oferta, `channels`, flags `offer_type_bogo` / `offer_type_discount`)
3. `inference_kmeans` atribui o cluster automaticamente
4. Se o cluster for o alvo configurado em `config.py` (padrão 1), `inference_dt` estima a probabilidade de aceite

## Estrutura do projeto

```
├── data/
│   ├── raw/              # JSONs de entrada
│   ├── processed/        # CSVs gerados pelo treino
│   └── images/           # Gráficos exploratórios, clusters e relatório da árvore
├── model/                # Modelos serializados (.pkl, gerados pelo treino)
├── src/
│   ├── config.py         # Caminhos, hiperparâmetros e constantes
│   ├── prep.py           # Limpeza e junção dos dados
│   ├── describe_ds.py    # Visualizações exploratórias e classification report
│   ├── clustering.py     # K-Means por perfil de usuário
│   └── decision_tree.py  # Classificação de aceite da oferta
├── run_train.py          # Pipeline de treino
└── run_forecast.py       # Inferência (UserProfile, UserOffer e funções de previsão)
```

## Como funciona

### 1. Preparação (`prep.py`)

Ofertas, perfis e transações são unidos em um único DataFrame. Cada linha é uma transação ligada à jornada da oferta (recebida, vista, completada). A coluna `offer_success` vale 1 se a oferta foi completada (`reward_jornada` preenchido), 0 caso contrário.

### 2. Clustering (`clustering.py`)

Por `account_id`, agrega-se idade, gênero, limite do cartão, valor médio de transação e taxa de sucesso. O K-Means usa apenas as features de perfil (`CLUSTERING_COLS_PROFILE`); a taxa de sucesso entra só no gráfico `clusters.png`. Artefatos: `model/kmeans.pkl`, `model/kmeans_scaler.pkl`, `model/kmeans_encoders.pkl`.

### 3. Árvore de decisão (`decision_tree.py`)

Treino **somente no cluster alvo** (padrão 1), prevendo `offer_success` a partir de oferta + perfil. Desbalanceamento tratado com **SMOTE**; hiperparâmetros via `GridSearchCV`. Modelo salvo em `model/decision_tree.pkl`.

### 4. Inferência (`run_forecast.py`)

1. `inference_kmeans` — cluster do perfil (carrega os três `.pkl` do K-Means)
2. `inference_dt` — probabilidades de não aceite / aceite (classe 0 e 1), alinhando colunas ao `feature_names_in_` do modelo treinado

## Configuração

Caminhos, cluster alvo, features e grid estão em `src/config.py`. Para mudar o cluster da árvore, altere `DECISION_TREE_TARGET_CLUSTER` e treine de novo.

## Saídas geradas

| Caminho | Descrição |
|---------|-----------|
| `data/processed/full_dataset.csv` | Dataset após prep (ainda com ofertas informacionais) |
| `data/processed/full_dataset_clustered.csv` | Dataset sem informacionais, com `cluster` e `taxa_sucesso` |
| `data/processed/test_dataset.csv` | Split de teste da árvore (cluster alvo) |
| `data/processed/classification_report.csv` | Métricas precision/recall/F1 no teste |
| `data/images/describe_offer.png` | Taxa de sucesso por atributo da oferta |
| `data/images/describe_profile.png` | Distribuição do perfil |
| `data/images/clusters.png` | Histogramas por cluster (inclui `taxa_sucesso`) |
| `data/images/describe_classification_report.png` | Heatmap do relatório de classificação |
| `model/kmeans.pkl` | Modelo K-Means |
| `model/kmeans_scaler.pkl` | `StandardScaler` do clustering |
| `model/kmeans_encoders.pkl` | `LabelEncoder` por coluna categórica |
| `model/decision_tree.pkl` | Árvore treinada (após one-hot no fit) |
