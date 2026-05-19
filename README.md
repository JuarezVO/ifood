# Previsão de aceite de ofertas

Pipeline de machine learning que agrupa usuários por perfil (K-Means) e prevê se uma oferta será aceita (árvore de decisão)

## Como utilizar

### Pré-requisitos

- Dados em `data/raw/` (`offers.json`, `profile.json`, `transactions.json`)

Ative o ambiente e instale as dependências:

```bash
pip install -r requirements.txt
```

### Treinar os modelos

Na raiz do projeto:

```bash
python run_train.py
```

O script:

1. Prepara e une os três JSONs
2. Gera gráficos exploratórios em `data/images/`
3. Agrupa usuários em clusters e salva os modelos em `model/`
4. Treina a árvore de decisão para o cluster alvo
5. Salva datasets processados em `data/processed/`

### Fazer previsões

Depois do treino, use as funções em `run_forecast.py`:
1. Edite os detalhes do usuário utilizando a classe UserProfile
2. Edite a oferta com a classe UserOffer
3. Defina o grupo do usuário através da análise de clusters
4. Se o grupo for 1, ele estimará o sucesso da oferta para o usuário

```bash
python run_forecast.py
```

## Estrutura do projeto

```
├── data/
│   ├── raw/              # JSONs de entrada
│   ├── processed/        # CSVs gerados pelo treino
│   └── images/           # Gráficos exploratórios e de clusters
├── model/                # Modelos serializados (.pkl)
├── src/
│   ├── config.py         # Caminhos, hiperparâmetros e constantes
│   ├── prep.py           # Limpeza e junção dos dados
│   ├── describe_ds.py    # Visualizações exploratórias
│   ├── clustering.py     # K-Means por perfil de usuário
│   └── decision_tree.py  # Classificação de aceite da oferta
├── run_train.py          # Pipeline de treino
└── run_forecast.py       # Inferência (perfil + oferta)
```

## Como funciona

### 1. Preparação (`prep.py`)

Os arquivos de ofertas, perfis e transações são unidos em um único dataset. Cada linha representa uma transação ligada à jornada da oferta (recebida, vista, completada). É criada a coluna `offer_success` (1 se a oferta foi utilizada, 0 caso contrário).

### 2. Clustering (`clustering.py`)

Por usuário (`account_id`), calcula-se um perfil agregado (idade, gênero, limite do cartão, valor médio de transação). O K-Means divide os usuários em **3 clusters**. Os artefatos ficam em `model/kmeans*.pkl`.

### 3. Árvore de decisão (`decision_tree.py`)

A árvore é treinada **apenas no cluster 1**, prevendo `offer_success` a partir de características da oferta e do perfil. Classes desbalanceadas são corrigidas com SMOTE; hiperparâmetros são escolhidos via GridSearchCV.

### 4. Inferência (`run_forecast.py`)

1. `inference_kmeans` — atribui o cluster ao perfil informado
2. `inference_dt` — para usuários do cluster alvo, retorna a probabilidade de 0 ou 1 (oferta não aceita / aceita)

## Configuração

Caminhos de arquivos, número de clusters, features, hiperparâmetros e demais constantes estão centralizados em `src/config.py`.

## Saídas geradas

| Caminho | Descrição |
|---------|-----------|
| `data/processed/full_dataset.csv` | Dataset completo após o prep |
| `data/processed/full_dataset_clustered.csv` | Dataset com coluna `cluster` |
| `data/images/*.png` | Gráficos exploratórios e de clusters |
| `model/kmeans.pkl` | Modelo de clustering |
| `model/kmeans_scaler.pkl` | Normalização usada no K-Means |
| `model/kmeans_encoders.pkl` | Encoders categóricos |
| `model/decision_tree.pkl` | Classificador de aceite |
