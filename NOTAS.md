# Notas de calibração sklearn ↔ Spark ML

Ambiente: `conda activate ml` → `python run_train.py`

## Prep (confirmado)

- `offer_success`: `reward_comp_jornada` nulo → 0, senão 1
- `discount_per_minvalue`: `try_divide(discount_value, min_value)`

## Clustering (2026-05-20)

| Cluster | Usuários (cluster_by_account.csv) |
|---------|-----------------------------------|
| 0 | 4635 |
| 1 | 4622 |
| 2 | 4219 |

- Gênero: `OrdinalEncoder` com ordem por **frequência** (equivalente ao `StringIndexer` Spark `frequencyDesc`)
- K-Means: `n_init="auto"`, `random_state=42`, `StandardScaler`

## Árvore — sklearn calibrado (cluster alvo = 1)

| Métrica | Valor |
|---------|-------|
| Linhas no cluster 1 | 227 647 |
| Treino: sucesso 0 / 1 | 8 866 / 173 251 |
| Acurácia teste | **0.6844** |
| Baseline | 95.12% |
| Modelo (precisão em pred=1) | 98.32% |
| Melhoria estimada | 3.36% |

**Top importâncias:** `credit_card_limit` (0.43), `age` (0.40), `discount_value` (0.08)

### Ajustes vs versão anterior

| Item | Alteração |
|------|-----------|
| `DECISION_TREE_TARGET_CLUSTER` | 1 (Spark original) |
| Split | sem `stratify` (como `randomSplit`) |
| Pesos | `sample_weight` fórmula Spark (`clf__sample_weight`) |
| Grid | `min_samples_split` [100, 150] em vez de `min_samples_leaf` |
| Categóricas | `OneHotEncoder` com categorias por frequência |

### Spark original (preencher se tiver log)

| Métrica | Spark | sklearn |
|---------|-------|---------|
| Acurácia teste | _?_| 0.6844 |
| Tamanho cluster 1 (usuários) | _?_| 4622 |
| Top-3 features | _?_| credit_card_limit, age, discount_value |

## Limitações

- K-Means sklearn ≠ Spark: clusters não são 1:1 entre implementações
- `maxBins` Spark não tem equivalente direto no sklearn
- Réplica exata exigiria pipeline Spark ML salvo para comparação de centróides
