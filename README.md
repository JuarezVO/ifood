# Previsão de aceite de ofertas

Pipeline de machine learning em **PySpark** que agrupa usuários por perfil (K-Means) e prevê se uma oferta será aceita (árvore de decisão).

## Como utilizar

### Pré-requisitos

- Ambiente conda `ml` com Python 3.13
- Java (JRE) instalado e `JAVA_HOME` configurado (Spark local)
- Dados em `data/raw/` (`offers.json`, `profile.json`, `transactions.json`)

Ative o ambiente e instale as dependências:

```bash
conda activate ml
pip install -r requirements.txt
```

### Treinar os modelos

Na raiz do projeto:

```bash
python run_train.py
```

O script:

1. Prepara e une os três JSONs com Spark DataFrames
2. Gera gráficos exploratórios em `data/images/`
3. Agrupa usuários em clusters e salva o pipeline em `model/kmeans_pipeline/`
4. Treina a árvore de decisão para o cluster alvo
5. Salva datasets processados em `data/processed/` (Parquet)

### Fazer previsões

Depois do treino, use as funções em `run_forecast.py`:

1. Edite os detalhes do usuário utilizando a classe `UserProfile`
2. Edite a oferta com a classe `UserOffer`
3. Execute o script para obter cluster e probabilidade de aceite

```bash
python run_forecast.py
```

## Estrutura do projeto

```
├── data/
│   ├── raw/              # JSONs de entrada
│   ├── processed/        # Parquet gerados pelo treino
│   └── images/           # Gráficos exploratórios e de clusters
├── model/                # Pipelines Spark ML
├── src/
│   ├── config.py         # Caminhos, hiperparâmetros e constantes
│   ├── spark_session.py  # Factory SparkSession local
│   ├── prep.py           # Limpeza e junção dos dados
│   ├── describe_ds.py    # Visualizações exploratórias
│   ├── clustering.py     # K-Means por perfil de usuário
│   └── decision_tree.py  # Classificação de aceite da oferta
├── run_train.py          # Pipeline de treino
└── run_forecast.py       # Inferência (perfil + oferta)
```

## Como funciona

### 1. Preparação (`prep.py`)

Os arquivos de ofertas, perfis e transações são unidos em um único Spark DataFrame. Cada linha representa uma transação ligada à jornada da oferta (recebida, vista, completada). É criada a coluna `offer_success` (1 se a oferta foi utilizada, 0 caso contrário).

### 2. Clustering (`clustering.py`)

Por usuário (`account_id`), calcula-se um perfil agregado (idade, gênero, limite do cartão, valor médio de transação). O K-Means divide os usuários em **3 clusters**. O pipeline Spark ML fica em `model/kmeans_pipeline/`.

### 3. Árvore de decisão (`decision_tree.py`)

A árvore é treinada **apenas no cluster 1**, prevendo `offer_success` a partir de características da oferta e do perfil. Classes desbalanceadas são tratadas com `weightCol`; hiperparâmetros são escolhidos via `CrossValidator`.

### 4. Inferência (`run_forecast.py`)

1. `inference_kmeans` — atribui o cluster ao perfil informado
2. `inference_dt` — para usuários do cluster alvo, retorna a probabilidade de 0 ou 1 (oferta não aceita / aceita)

## Configuração

Caminhos de arquivos, número de clusters, features, hiperparâmetros e demais constantes estão centralizados em `src/config.py`.

## Saídas geradas

| Caminho | Descrição |
|---------|-----------|
| `data/processed/full_dataset.parquet` | Dataset completo após o prep |
| `data/processed/full_dataset_clustered.parquet` | Dataset com coluna `cluster` |
| `data/images/*.png` | Gráficos exploratórios e de clusters |
| `model/kmeans_pipeline/` | Pipeline de clustering |
| `model/decision_tree_pipeline/` | Classificador de aceite |
