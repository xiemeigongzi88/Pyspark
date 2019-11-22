# Databricks notebook source
# sentiment analysis with pyspark 

# COMMAND ----------

# Sentiment Analysis with Pyspark
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline
import pandas as pd

# COMMAND ----------


from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# COMMAND ----------

from pyspark.sql import SQLContext
from pyspark import SparkContext

sqlContext = SQLContext(sc)

# COMMAND ----------

df = sqlContext.read.format('com.databricks.spark.csv').options(header='false', inferschema='true').load('/FileStore/tables/pyspark_learning/test/training_1600000_processed_noemoticon-efba6.csv')

# COMMAND ----------

pd.DataFrame(df.take(5), columns = df.columns).transpose()

# COMMAND ----------

df.show(5)

# COMMAND ----------

df = df.dropna()

# COMMAND ----------

df.columns

# COMMAND ----------

df.count()

# COMMAND ----------

(train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed = 2000)

# COMMAND ----------

# HashingTF + IDF + Logistic Regression

# COMMAND ----------

tokenizer = Tokenizer(inputCol='_c5', outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "_c0", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

# COMMAND ----------

train_set.show()

# COMMAND ----------

pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
train_df.show(5)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
accuracy

# COMMAND ----------

# CountVectorizer + IDF + Logistic Regression

# COMMAND ----------

# MAGIC %%time
# MAGIC from pyspark.ml.feature import CountVectorizer
# MAGIC 
# MAGIC tokenizer = Tokenizer(inputCol="_c5", outputCol="words")
# MAGIC cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
# MAGIC idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
# MAGIC label_stringIdx = StringIndexer(inputCol = "_c0", outputCol = "label")
# MAGIC lr = LogisticRegression(maxIter=100)
# MAGIC pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, lr])
# MAGIC 
# MAGIC pipelineFit = pipeline.fit(train_set)
# MAGIC predictions = pipelineFit.transform(val_set)
# MAGIC accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
# MAGIC roc_auc = evaluator.evaluate(predictions)
# MAGIC 
# MAGIC print("Accuracy Score: {0:.4f}".format(accuracy))
# MAGIC print("ROC-AUC: {0:.4f}".format(roc_auc))

# COMMAND ----------


