# Previsão de aceite de ofertas

Pipeline de machine learning com **PySpark** no ETL (prep, agregações, joins) e **scikit-learn** nos modelos (K-Means + árvore de decisão). Os modelos ficam **em memória** — treino e inferência ocorrem na mesma execução de `run_train.py`.

Para obter os mesmos resultados apresentados nos slides, utilize a branch principal.

```bash
git checkout master
```

## Como utilizar

### Pré-requisitos

- Python 3.13
- Java (JRE) instalado e `JAVA_HOME` configurado (Spark local)
- Dados em `data/raw/` (`offers.json`, `profile.json`, `transactions.json`)

No Windows, o Spark pode exigir `winutils.exe` / `hadoop.dll` em `tools/hadoop/bin` (pasta ignorada pelo git; configure `HADOOP_HOME` se necessário).

```bash
pip install -r requirements.txt
```

### Treinar e inferir

Na raiz do projeto:

```bash
python run_train.py
```

O script:

1. Prepara e une os três JSONs com Spark (`prep.py`)
2. Gera gráficos exploratórios em `data/images/` (`describe_ds.py`)
3. Remove ofertas informacionais, converte para pandas e salva `full_dataset.csv`
4. Agrupa usuários em 3 clusters (sklearn) e grava `cluster_by_account.csv`
5. Treina a árvore de decisão no cluster alvo e salva `full_dataset_clustered.csv`
6. Executa um exemplo de previsão com os modelos em memória

### Reutilizar as funções de inferência

Importe os objetos retornados pelo treino (ou variáveis de `run_train.py`) e use `src/forecast.py`:

```python
from src.forecast import inference_kmeans, inference_dt
from src.schemas import UserProfile, UserOffer

cluster = inference_kmeans(kmeans_model, scaler, encoders, user_profile)
proba = inference_dt(dt_model, user_profile, user_offer)  # se cluster == alvo
```

`run_forecast.py` apenas reexporta `UserProfile`, `UserOffer` e as funções acima. **Não há modelos em disco** — é preciso passar `kmeans_model`, `scaler`, `encoders` e `dt_model` da sessão de treino.

## Estrutura do projeto

```
├── data/
│   ├── raw/              # JSONs de entrada
│   ├── processed/        # CSVs gerados pelo treino (gitignored)
│   └── images/           # Gráficos exploratórios e de clusters
├── src/
│   ├── config.py         # Caminhos, hiperparâmetros e constantes
│   ├── schemas.py        # UserProfile, UserOffer (Pydantic)
│   ├── spark_session.py  # SparkSession local
│   ├── prep.py           # ETL Spark
│   ├── describe_ds.py    # Gráficos exploratórios
│   ├── clustering.py     # K-Means por perfil de usuário
│   ├── decision_tree.py  # Árvore de decisão (cluster alvo)
│   └── forecast.py       # Inferência com modelos em memória
├── run_train.py          # Pipeline completo + forecast de exemplo
└── run_forecast.py       # Reexporta schemas e funções de inferência
```

## Como funciona

### 1. Preparação (`prep.py`)

Ofertas, perfis e transações são unidos em um Spark DataFrame. Cada linha é uma transação ligada à jornada da oferta. `offer_success` vale 1 se `reward_comp_jornada` estiver preenchido, 0 caso contrário. `discount_per_minvalue` usa `try_divide(discount_value, min_value)`.

### 2. Clustering (`clustering.py`)

Por `account_id`, agrega idade, gênero, limite, valor médio de transação (`avg(amount)`) e taxa de sucesso. O K-Means usa `CLUSTERING_COLS_PROFILE`; `taxa_sucesso` entra só no gráfico `clusters.png`. Gênero é codificado com `LabelEncoder` (ordem alfabética das classes).

### 3. Árvore de decisão (`decision_tree.py`)

Treino **somente no cluster alvo** (`DECISION_TREE_TARGET_CLUSTER`, padrão **1**). Prevê `offer_success` a partir de oferta + perfil. Pesos de classe seguem a fórmula do Spark ML (`clf__sample_weight`); split de teste **sem** estratificação; grid em `max_depth` e `min_samples_split`; categorias do `OneHotEncoder` ordenadas por frequência no treino.

### 4. Inferência (`forecast.py`)

1. `inference_kmeans` — aplica encoders + scaler + K-Means
2. `inference_dt` — probabilidades de não aceite (0) e aceite (1) via pipeline sklearn

## Configuração

Constantes em `src/config.py`. Para mudar o cluster da árvore, altere `DECISION_TREE_TARGET_CLUSTER` e rode o treino de novo.

## Saídas geradas

| Caminho | Descrição |
|---------|-----------|
| `data/processed/full_dataset.csv` | Dataset em pandas sem ofertas informacionais |
| `data/processed/cluster_by_account.csv` | Mapeamento `account_id` → `cluster` |
| `data/processed/full_dataset_clustered.csv` | Dataset com `cluster` e `taxa_sucesso` |
| `data/images/describe_offer.png` | Taxa de sucesso por atributo da oferta |
| `data/images/describe_profile.png` | Perfil (gênero, limite, idade, valor) |
| `data/images/clusters.png` | Histogramas por cluster (inclui `taxa_sucesso`) |

Gráficos exploratórios refletem o DataFrame **antes** do filtro de ofertas informacionais; os CSVs de treino são gerados **depois** desse filtro.
