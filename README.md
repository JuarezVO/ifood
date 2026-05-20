# Previsão de aceite de ofertas

Pipeline de machine learning que usa **PySpark** no ETL e **scikit-learn** nos modelos (K-Means + árvore de decisão). Os modelos ficam **em memória** — treino e inferência ocorrem na mesma execução.

## Como utilizar

### Pré-requisitos

- Ambiente conda `ml` com Python 3.13
- Java (JRE) instalado e `JAVA_HOME` configurado (Spark local, apenas ETL)
- Dados em `data/raw/` (`offers.json`, `profile.json`, `transactions.json`)

Ative o ambiente e instale as dependências:

```bash
conda activate ml
pip install -r requirements.txt
```

### Treinar e inferir

Na raiz do projeto:

```bash
conda activate ml
python run_train.py
```

O script:

1. Prepara e une os três JSONs com Spark DataFrames
2. Gera gráficos exploratórios em `data/images/`
3. Agrupa usuários em clusters (sklearn)
4. Treina a árvore de decisão para o cluster alvo
5. Salva datasets processados em `data/processed/` (CSV)
6. Executa um exemplo de previsão com os modelos em memória

### Reutilizar as funções de inferência

Importe os pipelines retornados pelo treino e use `src/forecast.py`:

```python
from src.forecast import inference_kmeans, inference_dt
from src.schemas import UserProfile, UserOffer

cluster = inference_kmeans(kmeans_model, user_profile)
proba = inference_dt(dt_model, user_profile, user_offer)
```

`run_forecast.py` expõe essas funções, mas **não carrega modelos do disco** — é necessário passar os pipelines treinados.

## Estrutura do projeto

```
├── data/
│   ├── raw/              # JSONs de entrada
│   ├── processed/        # CSV gerados pelo treino
│   └── images/           # Gráficos exploratórios e de clusters
├── src/
│   ├── config.py         # Caminhos, hiperparâmetros e constantes
│   ├── schemas.py        # UserProfile, UserOffer
│   ├── forecast.py       # Inferência (modelos em memória)
│   ├── spark_session.py  # Factory SparkSession local (ETL)
│   ├── prep.py           # Limpeza e junção dos dados
│   ├── describe_ds.py    # Visualizações exploratórias
│   ├── clustering.py     # K-Means por perfil de usuário
│   └── decision_tree.py  # Classificação de aceite da oferta
├── run_train.py          # Pipeline de treino + forecast de exemplo
└── run_forecast.py       # Reexporta funções de inferência
```

## Como funciona

### 1. Preparação (`prep.py`)

Os arquivos de ofertas, perfis e transações são unidos em um único Spark DataFrame. Cada linha representa uma transação ligada à jornada da oferta (recebida, vista, completada). É criada a coluna `offer_success` (1 se a oferta foi utilizada, 0 caso contrário).

### 2. Clustering (`clustering.py`)

Por usuário (`account_id`), calcula-se um perfil agregado (idade, gênero, limite do cartão, valor médio de transação). O K-Means (sklearn) divide os usuários em **3 clusters** e retorna o pipeline treinado em memória.

### 3. Árvore de decisão (`decision_tree.py`)

A árvore é treinada **apenas no cluster alvo** (`DECISION_TREE_TARGET_CLUSTER`, padrão 1), prevendo `offer_success` a partir de características da oferta e do perfil. Pesos de classe seguem a fórmula do Spark ML (`sample_weight`); hiperparâmetros são escolhidos via `GridSearchCV` (`min_samples_split` alinhado a `minInstancesPerNode`).

> **Nota:** O pipeline sklearn foi calibrado para **aproximar** o Spark ML original. K-Means (sklearn vs Spark) não replica clusters idênticos; métricas podem divergir alguns pontos percentuais.

### 4. Inferência (`forecast.py`)

1. `inference_kmeans` — atribui o cluster ao perfil informado
2. `inference_dt` — para usuários do cluster alvo, retorna a probabilidade de 0 ou 1 (oferta não aceita / aceita)

## Configuração

Caminhos de arquivos, número de clusters, features, hiperparâmetros e demais constantes estão centralizados em `src/config.py`.

## Saídas geradas

| Caminho | Descrição |
|---------|-----------|
| `data/processed/full_dataset.csv` | Dataset completo após o prep (sem ofertas informacionais) |
| `data/processed/cluster_by_account.csv` | Mapeamento account_id → cluster |
| `data/processed/full_dataset_clustered.csv` | Dataset com coluna `cluster` |
| `data/images/*.png` | Gráficos exploratórios e de clusters |
