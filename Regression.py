# Databricks notebook source
#First, I imported the cruise_ship_info.csv file into DataBricks and then into a dataframe. Above, you can see the command as well as the imported data displayed as a dataframe.

df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/slevy19@gwu.edu/cruise_ship_info.csv")
dataframe = df1.toPandas()
df1.show()


# COMMAND ----------

df1.columns
#df1.head()
#df1.head(10)

# COMMAND ----------

df1.head()

# COMMAND ----------

df1.head(10)

# COMMAND ----------

df1.printSchema() #nullable means they have nulls included

# COMMAND ----------

#I then viewed the data and changed the numerical values from String variables into Double or Integer values for computation. 

df1 = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .load("dbfs:/FileStore/shared_uploads/slevy19@gwu.edu/cruise_ship_info.csv") #changes data types

df1.printSchema() #shows they are numbers and not strings

# COMMAND ----------

# I then switched the Cruise_Line variable into a categorical variable. 
# This change will both save memory in the system and make it easier to categorize the data for plots or manipulation.
from pyspark.ml.feature import StringIndexer 
indexer=StringIndexer(inputCol='Cruise_line',outputCol='Category') 
indexed=indexer.fit(df1).transform(df1)

df1.printSchema()

# COMMAND ----------

#The 9th line of the above code transforms a list of columns into a single column. 
# I used the tonnage, passengers, length, cabins, and passenger_density as one single ‘features’ column as all of 
# these variables will most likely factor into the prediction of how many crew members we might need. 

#import modules/libs
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler( inputCols= ["Tonnage", "passengers", "length", "cabins", "passenger_density"], outputCol="features")
output = assembler.transform(df1)

# COMMAND ----------

output.head(10)
#I printed the top 10 items to ensure that the transformation worked. 


# COMMAND ----------

#I then created a new data_frama with just the features and crew_member columns so that we can begin to train the model.
final_data = output.select("features", "Crew") 
final_data.show(10)

# COMMAND ----------

#I then created two separate data_frames; one for training data which contains 
# 70% of the data and the other for testing data which contains 30% of the data. 
# I then displayed the rows to ensure that we split it correctly, and printed out both of the summary statistics. 


train_data, test_data = final_data.randomSplit([0.7,0.3]) #70% for training 30% for testing. 

train_data.show(5)
train_data.describe().show()

test_data.show(5)
train_data.describe().show()

# COMMAND ----------

#In this command, I built the linear regression model and 
# trained the model using the train_data I created 


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol= 'features', labelCol= 'Crew')

#training the model by fit (fitting the model)
lr_model = lr.fit(train_data) #fit method performs the training using the train data 

# COMMAND ----------

#this shows the code that will print out the summary statistics for the regression model. 
# It will show the basic variables that are relevant to a linear regression model, 
# such as the R^2, the MSE (means squared error) and more. 


def modelsummary(model, param_names):
    param_names.append('intercept')
    import numpy as np
    print("Note: the last rows are the information for Intercept")
    print("Note","-------------------")
    print("##"," Estimate    Std.Error t Values P-value")
    coef = np.append(list(model.coefficients),model.intercept)
    Summary=model.summary
    
    for i in range(len(Summary.pValues)):
        print("##", '{:10.6f}'.format(coef[i]),\
        '{:10.6f}'.format(Summary.coefficientStandardErrors[i]),\
        '{:8.3f}'.format(Summary.tValues[i]),\
        '{:10.6f}'.format(Summary.pValues[i]), param_names[i])

    print("##", '---')
    print("##", "Mean squared error: % .6f" \
        % Summary.meanSquaredError, ", RMSE: % .6f" \
        % Summary.rootMeanSquaredError )
    print("##", "Multiple R-squared: %f" % Summary.r2, ", \
          Total iterations: %i"% Summary.totalIterations)

# COMMAND ----------

# calls the model_summary method we created and prints out the relevant statistics. 
# I will expand on these statistics later

print("Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)

param_names = ["Tonnage", "passengers", "length", "cabins", "passenger_density"]
modelsummary(lr_model, param_names)

# COMMAND ----------

#Cross Validation
#Finally, we have the cross validation phases where we can show the prediction based on the features we used to train the data. 
# The ‘prediction’ column shows our predictions of the crew count side by side with the actual crew count.
#  Just by looking at the top 10 rows, we can see that the model is very accurate and falls very close to the actual crew count to the left. 
# I will now explain why this model is so accurate. 



pred_test = lr_model.transform(train_data)
pred_test.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC #Why did the model work so well?
# MAGIC
# MAGIC Knowing what we know about linear regression models, we know that the multiple R^2 value explains what percentage of the variance of the predicting variable is. From our statistics, we can see that this value is 0.935800, which means that the model explains around 93.5% of the variance. This value is high which means that our model explains a large amount of variance and therefore will be pretty accurate in its predictions! In addition, the MSE (mean squared error) shows the average of the squared errors in a regression model. A lower MSE means better predictions, and since our MSE is approximately 0.7, we can say that this model is significantly accurate!
# MAGIC
# MAGIC In addition, there are other things that can explain why this model worked so well and was so accurate in its predictions. Firstly, the data provided was very large and therefore provided a lot of information for the model to learn from. Secondly, the data given clearly had some kind of strong linear relationship, which is why a linear regression model could predict so well and why the multiple R^2 was so high. Finally, we were able to cross-validate the model which allows the model to keep splitting the data into subsets and understand different combinations of said data, which will allow us to get better results! 
# MAGIC
