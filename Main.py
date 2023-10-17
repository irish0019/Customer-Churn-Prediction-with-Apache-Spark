from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from IPython.display import display
import pandas as pd
spark = SparkSession.builder.appName("CustomerChurnPrediction").getOrCreate()
data = spark.read.csv('/content/drive/MyDrive/data_preview', header=True, inferSchema=True)
data = data.drop("unwanted_columns")
assembler = VectorAssembler(inputCols=['usage_months','calls_made'], outputCol="features")
train_data, test_data = data.randomSplit([0.7, 0.3])
lr = LogisticRegression(labelCol="churn_label", featuresCol="features")
pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(train_data)
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="churn_label", rawPredictionCol="rawPrediction")
areaUnderROC = evaluator.evaluate(predictions)
print("Area Under ROC: ", areaUnderROC)
new_data = spark.read.csv('/content/drive/MyDrive/CSE412DATA.csv', header=True, inferSchema=True)
new_predictions = model.transform(new_data)
new_predictions.show()
spark.stop()
