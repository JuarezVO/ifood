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
    transactions = transactions.select(
        "account_id",
        "event",
        "time_since_test_start",
        col("value.amount").alias("amount"),
        when(col("value.`offer id`").isNotNull(), col("value.`offer id`"))
        .otherwise(col("value.offer_id")).alias("offer id"),
        col("value.reward").alias("reward"),
    )

    recebido = transactions.filter(col("event") == "offer received")
    visto = transactions.filter(col("event") == "offer viewed")
    completado = transactions.filter(col("event") == "offer completed")
    trans = transactions.filter(col("event") == "transaction")

    window_seq = Window.partitionBy("account_id", "offer id").orderBy("time_since_test_start")
    recebido = recebido.withColumn("seq", row_number().over(window_seq))
    visto = visto.withColumn("seq", row_number().over(window_seq))
    completado = completado.withColumn("seq", row_number().over(window_seq))

    recebido_visto = _suffix_columns(recebido, "_recv", {"account_id", "offer id", "seq"}).join(
        _suffix_columns(visto, "_view", {"account_id", "offer id", "seq"}),
        on=["account_id", "offer id", "seq"],
        how="left",
    )

    recebido_visto_completado = recebido_visto.join(
        _suffix_columns(completado, "_comp", {"account_id", "offer id", "seq"}),
        on=["account_id", "offer id", "seq"],
        how="left",
    ).drop("seq")

    jornada_exclude = {"account_id"}
    jornada = _suffix_columns(recebido_visto_completado, "_jornada", jornada_exclude)
    jornada = jornada.withColumnRenamed("offer id", "offer id_jornada")

    return trans.join(jornada, on="account_id", how="left")


def prep_datasets(
    spark: SparkSession,
    offers_path: str,
    profiles_path: str,
    transactions_path: str,
) -> DataFrame:
    print("Preparando datasets em ambiente Spark\n")

    offers = spark.read.json(offers_path)
    profiles = spark.read.json(profiles_path)
    transactions = _prep_transactions(spark.read.json(transactions_path))

    df = transactions.join(
        offers,
        transactions["offer id_jornada"] == offers["id"],
        how="left",
    ).drop(offers["id"])

    df = df.join(
        profiles,
        df["account_id"] == profiles["id"],
        how="left",
    ).drop(profiles["id"])

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
    df = df.withColumn("offer id", col("offer id_jornada"))
    df = df.withColumn("offer_code", _offer_code_column())

    # CORREÇÃO 1: mean_amount_per_account calculado ANTES do filter de idade e dropna
    # (igual ao sklearn: linha 52 antes das linhas 53 e 56)
    window_mean = Window.partitionBy("account_id")
    df = df.withColumn("mean_amount_per_account", avg("amount").over(window_mean))

    # CORREÇÃO 2: filter de idade DEPOIS da média
    df = df.filter(col("age") <= PREP_AGE_MAX)

    # CORREÇÃO 3: dropna DEPOIS da média e do filter
    df = df.dropna(subset=PREP_DROP_NA_SUBSETS)

    for old, new in PREP_COLS_TO_RENAME.items():
        df = df.withColumnRenamed(old, new)

    df = df.drop(*[column for column in PREP_COLS_TO_DROP if column in df.columns])

    return df
