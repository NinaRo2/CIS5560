# Databricks notebook source
# Import Spark SQL and Spark ML libraries

from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel, DecisionTreeRegressionModel
from pyspark.ml.feature import StringIndexer


IS_DB = True

# COMMAND ----------



# COMMAND ----------

PYSPARK_CLI = False
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------



# COMMAND ----------

tripSchema = StructType([
  StructField('TripID', StringType()),
  StructField('TripStartTimestamp', StringType()),
  StructField('TripEndTimestamp', StringType()),
  StructField('TripSeconds', IntegerType()),
  StructField('TripMiles', FloatType()),
  StructField('PickupCommunityArea', StringType()),
  StructField('DropoffCommunityArea', StringType()),
  StructField('Fare', FloatType()),
  StructField('PickupCentroidLatitude', StringType()),
  StructField('PickupCentroidLongitude', StringType()),
  StructField('DropoffCentroidLatitude', StringType()),
  StructField('DropoffCentroidLongitude', StringType()),
  StructField('AvgCostMile', FloatType()),
  StructField('DayofWeek', FloatType())
  ])

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/trips_sample.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .schema(tripSchema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
  
display(df)

# COMMAND ----------

# Create a view or table
temp_table_name = "trips_sample_csv"
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

if PYSPARK_CLI:
    csv = spark.read.csv('trips_sample.csv', inferSchema=True, header=True)
else:
    csv = spark.sql("SELECT * FROM trips_sample_csv")


csv.show(5)

# COMMAND ----------

df1=df.dropna()

# COMMAND ----------

data = df1.select('TripMiles','TripSeconds', col('Fare').alias('label'))

# COMMAND ----------

data.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Split the Data

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Gradient Boosted Decision Tree Regression

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['TripMiles','TripSeconds'], outputCol="features")
gbt = GBTRegressor(labelCol="label")


# COMMAND ----------

# MAGIC %md
# MAGIC ####Tune Parameters

# COMMAND ----------

paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()
  
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.

#mlflow.sklearn.autolog()
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
#with mlflow.start_run():

# COMMAND ----------

# MAGIC %md
# MAGIC ####Define the Pipeline

# COMMAND ----------

pipeline = Pipeline(stages=[assembler, cv])
pipelineModel = pipeline.fit(train)

# COMMAND ----------

predictions = pipelineModel.transform(test)

# COMMAND ----------

predicted = predictions.select("features", "prediction", "trueLabel")
predicted.show(100)

# COMMAND ----------

predicted.createOrReplaceTempView("regressionPredictions")


# COMMAND ----------

dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")

display(dataPred)

# COMMAND ----------

evaluator  = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE on our test set: %g" % rmse)
#print "Root Mean Square Error (RMSE) for GBT Regression :", rmse

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['Job_ID','Posting_Type','Title_Code_No','Level','FullTime_PartTime','Salary_Frequency','Hours_Shift'], outputCol="features")
lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)
pipeline1 = Pipeline(stages=[assembler, lr])

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="r2")

r2 = evaluator.evaluate(predictions)

print("R2 (r2):", r2)
