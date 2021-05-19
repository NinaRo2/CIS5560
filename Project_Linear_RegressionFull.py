# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import abs, to_timestamp
from pyspark import SparkContext, SparkConf, SQLContext
from datetime import datetime
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector

from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

IS_DB = True

# COMMAND ----------

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

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

# File location and type
file_location = "/user/nrobert/trips_sample.csv"
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
  
#display(df)
df.show(10)
# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 2: Create a temporary view of the dataframe df

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

data = spark.sql("SELECT * FROM trips_sample_csv") #df_data_1

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

lr = LinearRegression(maxIter=5, regParam=0.3, elasticNetParam=0.8)

# COMMAND ----------

assembler = VectorAssembler(inputCols = ["TripMiles", "TripSeconds"], outputCol="features")

# COMMAND ----------

training = assembler.transform(train).select(col("features"), (col("Fare").cast("Double").alias("label")))
#training.show()

# COMMAND ----------

lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=5, regParam=0.3)
model = lr.fit(training)
print("Model trained!")

# COMMAND ----------

# Fit the model
lrModel = lr.fit(training)

# COMMAND ----------

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# COMMAND ----------

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
#print("rae: %f" % trainingSummary.rae)
#print("MSE: %s, MAE: %s, R2: %s" %(r_sq_sum/n, r_abs_sum/n, 1-r_sq_sum/y_var_sum))

# COMMAND ----------

print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

