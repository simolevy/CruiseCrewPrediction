# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

from pyspark.sql.functions import when
# File location and type
file_location = "/FileStore/tables/farm_ads-1.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df = df.withColumn('_c0', when(df['_c0'] == -1, 0).otherwise(1))

display(df)

# COMMAND ----------

from pyspark.sql.functions import col

# convert the column c0 to integer.
df = df.withColumn('_c0', col('_c0').cast('int'))
display(df)


# COMMAND ----------

 from pyspark.sql.functions import regexp_replace

#use the regexp_replace function to remove the word ad- from all of the columns

df = df.withColumn('_c0', regexp_replace(df['_c0'], 'ad-', ''))
df = df.withColumn('_c1', regexp_replace(df['_c1'], 'ad-', ''))

#rename the cols to work with them easier
df = df.withColumnRenamed('_c0', 'relevance')
df = df.withColumnRenamed('_c1', 'ad')

df.show()



# COMMAND ----------

#preparing the catagorical data
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

#map all of the columns out into numbers
categorical_columns = df.columns[0:2]
stage_string = [StringIndexer(inputCol = c, outputCol= c + "_string_encoded") for c in categorical_columns]

stage_one_hot = [OneHotEncoder(inputCol = c + "_string_encoded", outputCol = c + "_one_hot") for c in categorical_columns]

#execute the process!
ppl = Pipeline(stages=stage_string + stage_one_hot)
df = ppl.fit(df).transform(df)

# COMMAND ----------


from pyspark.sql.functions import col

#convert the relevance column into an integer so we can later apply the logistic regression.
df = df.withColumn('relevance', col('relevance').cast('int'))
assembler = VectorAssembler(
    inputCols = ['relevance_one_hot',
                 'ad_one_hot'],
    outputCol="features"
)

display(df)

#create a new label for the relevance column ( the one we are trying to predict )
farm_df = assembler.transform(df)
farm_df = farm_df.withColumn('label', F.col('relevance'))
farm_df.select('features', 'label').limit(10).show()

# COMMAND ----------

#split the data 60-40 into training and test with a random seed of 123.
training, test = farm_df.randomSplit([0.6,0.4], seed = 123)


# COMMAND ----------

#import the logistic regression 
from pyspark.ml.regression import GeneralizedLinearRegression

#apply the log regression model

logr = GeneralizedLinearRegression(family="binomial", link="logit", regParam=0.0)

logr_Model = logr.fit(training)

# COMMAND ----------

#create the model summary that will show us the testing/training outputs of our log regression.
def modelsummary(model, param_names):
    import numpy as np
    print("Note: the last rows are the info for Intercept")
    print('##', " Estimate   |     Std.error   | t-Vals    |   P-value")
    coef = np.append(list(model.coefficients), model.intercept)
    Summary = model.summary
    param_names.append('intercept')

    for i in range(len(Summary.pValues)):
        print('##', '{:10.6f}'.format(coef[i]),\
            '{:14.6f}'.format(Summary.coefficientStandardErrors[i]),\
                '{:12.3f}'.format(Summary.tValues[i]),\
                    '{12.6f}'.format(Summary.pValues[i]),\
                        param_names[i])
        #name the parameters to coincide with what we are trying to predict.
        param_names = ['relevant',
                    'non_relevant']
        
        modelsummary(logr_Model, param_names)

# COMMAND ----------

#show the prediction on the training data.
pred_training_cv = logr_Model.transform(training)
pred_training_cv.show(5, truncate = True)

# COMMAND ----------

# show the prediction on the test data
pred_test_cv = logr_Model.transform(test)
pred_test_cv.show(5, truncate = True)

# COMMAND ----------

# create a confusion matrix!

from sklearn.metrics import confusion_matrix

# show the true outcomes that the model is predicting, create a new pred_label col and classify the predictions..
y_true = pred_test_cv.select("label")
y_true = y_true.toPandas()
y_pred = pred_test_cv.select("prediction").withColumn('pred_label', F.when(F.col('prediction') > 0.5, 1).otherwise(0))
y_pred = y_pred.select('pred_label').toPandas()
cnf_matrix = confusion_matrix(y_true, y_pred)

#define the labels for the matrix.
import seaborn as sns
xlabels = ["Predicted:NO", "Predicted:YES"]
ylabels = ["Actual:NO", "Actual:YES"]

#visualize the matrix as a heatmap.
sns.heatmap (data = cnf_matrix,
             xticklabels = xlabels,
             yticklabels = ylabels,
             annot=True, fmt='g', cmap='crest')

# COMMAND ----------

# results

# based on the results of the confusion matrix, it seems as if there are no (0) false positives or false negatives, which indicates that 
# the model is working incredibly well at predicting the relevance of the advertisements! this could be because of the training split that we did
# as well as the size and overall quality of the data set.
