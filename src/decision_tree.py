from time import perf_counter

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when

from src.config import (
    DECISION_TREE_CATEGORICAL_FEATURES,
    DECISION_TREE_GRID_CV,
    DECISION_TREE_GRID_PARAMS,
    DECISION_TREE_NUMERIC_FEATURES,
    DECISION_TREE_TARGET_CLUSTER,
    DECISION_TREE_TARGET_COL,
    DECISION_TREE_TEST_SIZE,
    MODEL_DECISION_TREE_PIPELINE_PATH,
    RANDOM_STATE,
)

ts = perf_counter()


def _add_class_weights(df: DataFrame) -> DataFrame:
    counts = df.groupBy(DECISION_TREE_TARGET_COL).agg(count("*").alias("cnt")).collect()
    total = sum(row["cnt"] for row in counts)
    weight_map = {row[DECISION_TREE_TARGET_COL]: total / (len(counts) * row["cnt"]) for row in counts}
    weight_expr = None
    for label, weight in weight_map.items():
        condition = when(col(DECISION_TREE_TARGET_COL) == label, weight)
        weight_expr = condition if weight_expr is None else weight_expr.when(
            col(DECISION_TREE_TARGET_COL) == label, weight
        )
    return df.withColumn("weight", weight_expr)


def train_decision_tree(df: DataFrame) -> None:
    print("Decision Tree\n")
    print("Tamanho dos clusters:")
    df.groupBy("cluster").count().orderBy("cluster").show()

    cluster_df = df.filter(col("cluster") == DECISION_TREE_TARGET_CLUSTER)
    cluster_df = _add_class_weights(cluster_df)

    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_idx",
            handleInvalid="keep",
        )
        for column in DECISION_TREE_CATEGORICAL_FEATURES
    ]
    encoders = [
        OneHotEncoder(
            inputCol=f"{column}_idx",
            outputCol=f"{column}_ohe",
            handleInvalid="keep",
        )
        for column in DECISION_TREE_CATEGORICAL_FEATURES
    ]
    feature_cols = DECISION_TREE_NUMERIC_FEATURES + [
        f"{column}_ohe" for column in DECISION_TREE_CATEGORICAL_FEATURES
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    classifier = DecisionTreeClassifier(
        labelCol=DECISION_TREE_TARGET_COL,
        featuresCol="features",
        weightCol="weight",
        seed=RANDOM_STATE,
    )
    pipeline = Pipeline(stages=[*indexers, *encoders, assembler, classifier])

    train_df, test_df = cluster_df.randomSplit(
        [1 - DECISION_TREE_TEST_SIZE, DECISION_TREE_TEST_SIZE],
        seed=RANDOM_STATE,
    )

    print("Dataset treino:")
    train_df.groupBy(DECISION_TREE_TARGET_COL).count().show()

    param_grid = ParamGridBuilder().addGrid(
        classifier.maxDepth, DECISION_TREE_GRID_PARAMS["maxDepth"]
    ).addGrid(
        classifier.minInstancesPerNode, DECISION_TREE_GRID_PARAMS["minInstancesPerNode"]
    ).addGrid(
        classifier.maxBins, DECISION_TREE_GRID_PARAMS["maxBins"]
    ).build()

    evaluator = MulticlassClassificationEvaluator(
        labelCol=DECISION_TREE_TARGET_COL,
        predictionCol="prediction",
        metricName="accuracy",
    )
    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=DECISION_TREE_GRID_CV,
        seed=RANDOM_STATE,
    )

    cv_model = cross_validator.fit(train_df)
    best_model = cv_model.bestModel
    predictions = best_model.transform(test_df)

    accuracy = evaluator.evaluate(predictions)
    print(f"Acurácia no teste: {accuracy:.4f}")

    dt_model = best_model.stages[-1]
    importances = dt_model.featureImportances.toArray()
    for feature, importance in sorted(
        zip(feature_cols, importances),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{feature}: {importance:.4f}")

    baseline = cluster_df.agg({DECISION_TREE_TARGET_COL: "avg"}).collect()[0][0]
    positive_precision = (
        predictions.filter(col("prediction") == 1)
        .groupBy("prediction")
        .count()
        .collect()
    )
    total_positive_preds = sum(row["count"] for row in positive_precision) if positive_precision else 0
    true_positive = predictions.filter(
        (col("prediction") == 1) & (col(DECISION_TREE_TARGET_COL) == 1)
    ).count()
    modelo = true_positive / total_positive_preds if total_positive_preds else 0.0
    ganho = (modelo - baseline) / baseline * 100 if baseline else 0.0
    print(
        f"Base: {baseline * 100.0:.2f}% | Modelo: {modelo * 100.0:.2f}% | "
        f"Melhoria estimada: {ganho:.2f}%\n"
    )

    best_model.write().overwrite().save(MODEL_DECISION_TREE_PIPELINE_PATH)
    print(f"Modelo salvo em {MODEL_DECISION_TREE_PIPELINE_PATH}\n")
    print(f"Time: {round((perf_counter() - ts) / 60, 2)}min")
