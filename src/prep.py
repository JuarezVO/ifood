from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    concat_ws,
    substring,
    to_date,
    transform,
    try_divide,
    when,
    row_number,
)
from pyspark.sql.window import Window

from src.config import (
    PREP_AGE_MAX,
    PREP_COLS_TO_DROP,
    PREP_COLS_TO_RENAME,
    PREP_DROP_NA_SUBSETS,
    PREP_OFFER_CODE,
)


def _suffix_columns(df: DataFrame, suffix: str, exclude: set[str]) -> DataFrame:
    return df.select([
        col(column).alias(f"{column}{suffix}") if column not in exclude else col(column)
        for column in df.columns
    ])


def _offer_code_column():
    expr = None
    for offer_id, label in PREP_OFFER_CODE.items():
        condition = when(col("`offer id`") == offer_id, label)
        expr = condition if expr is None else expr.when(col("`offer id`") == offer_id, label)
    return expr.otherwise(None)


def _prep_transactions(transactions: DataFrame) -> DataFrame:
    # Seleciona e normaliza as colunas do JSON
    transactions = transactions.select(
        "account_id",
        "event",
        "time_since_test_start",
        col("value.amount").alias("amount"),
        col("value.`offer id`").alias("offer id"),
        col("value.offer_id").alias("offer_id"),
        col("value.reward").alias("reward"),
    )

    # Separa os eventos
    recebido = transactions.filter(col("event") == "offer received").drop("offer_id")
    visto = transactions.filter(col("event") == "offer viewed").drop("offer_id")
    completado = (
        transactions.filter(col("event") == "offer completed")
        .drop("offer id")
        .withColumnRenamed("offer_id", "offer id")
    )
    trans = transactions.filter(col("event") == "transaction").drop("offer_id", "offer id")

    # --- CRITICAL FIX FOR SPARK JOIN DUPLICATION ---
    # Janela para garantir que se houver ofertas repetidas para o mesmo ID,
    # o Spark ordene por tempo e não faça produto cartesiano desordenado.
    window_dedup = Window.partitionBy("account_id", "offer id").orderBy("time_since_test_start")

    recebido = recebido.withColumn("seq", row_number().over(window_dedup))
    visto = visto.withColumn("seq", row_number().over(window_dedup))
    completado = completado.withColumn("seq", row_number().over(window_dedup))

    # Realiza os joins incluindo a sequência temporal ("seq") para espelhar o Pandas
    recebido_visto = _suffix_columns(recebido, "_recv", {"account_id", "offer id", "seq"}).join(
        _suffix_columns(visto, "_view", {"account_id", "offer id", "seq"}),
        on=["account_id", "offer id", "seq"],
        how="left",
    )

    recebido_visto_completado = recebido_visto.join(
        _suffix_columns(completado, "_comp", {"account_id", "offer id", "seq"}),
        on=["account_id", "offer id", "seq"],
        how="left",
    ).drop("seq") # Remove a coluna de controle após os joins

    jornada_suffixes = {"account_id", "offer id", "event", "time_since_test_start", "amount", "reward"}
    jornada = _suffix_columns(recebido_visto_completado, "_jornada", jornada_suffixes)

    return trans.join(jornada, on="account_id", how="left")


def prep_datasets(
    spark: SparkSession,
    offers_path: str,
    profiles_path: str,
    transactions_path: str,
) -> DataFrame:
    print("Preparando datasets (Versão Corrigida)\n")

    # Leitura dos dados originais
    offers = spark.read.json(offers_path)
    profiles = spark.read.json(profiles_path)
    transactions = _prep_transactions(spark.read.json(transactions_path))

    # Joins com tabelas de apoio
    df = transactions.join(
        offers,
        transactions["offer id"] == offers["id"],
        how="left",
    ).drop(offers["id"])

    df = df.join(
        profiles,
        df["account_id"] == profiles["id"],
        how="left",
    ).drop(profiles["id"])

# Transformações de colunas e engenharia de features básicas
    df = df.withColumn("registered_on", to_date(col("registered_on"), "yyyyMMdd"))
    df = df.withColumn(
        "offer_success",
        when(col("reward_comp_jornada").isNull(), 0).otherwise(1),
    )
    df = df.withColumn("discount_per_minvalue", try_divide(col("discount_value"), col("min_value")))
    df = df.withColumn(
        "channels",
        concat_ws("", transform(col("channels"), lambda c: substring(c, 1, 1))),
    )

    # --- PASSO ADICIONADO AQUI: Criar o offer_code ANTES dos filtros ---
    df = df.withColumn("offer_code", _offer_code_column())

    # --- ORDEM DOS FILTROS MANTIDA ---
    # 1. Primeiro limpamos a base deletando idades inválidas e os nulos (Agora o offer_code existe aqui!)
    df = df.filter(col("age") <= PREP_AGE_MAX)
    df = df.dropna(subset=PREP_DROP_NA_SUBSETS)

    # 2. Com a base limpa, calculamos a média por conta (Garante o alinhamento com o Pandas)
    window_mean = Window.partitionBy("account_id")
    df = df.withColumn("mean_amount_per_account", avg("amount").over(window_mean))

    # Renomeia e dropa as colunas finais de acordo com a configuração
    for old, new in PREP_COLS_TO_RENAME.items():
        df = df.withColumnRenamed(old, new)

    df = df.drop(*[column for column in PREP_COLS_TO_DROP if column in df.columns])

    return df
