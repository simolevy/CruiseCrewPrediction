# Advertisement Relevance Prediction Model

Dataset --> https://www.kaggle.com/datasets/devvret/farm-ads-binary-classification

In this project, I used the 'relevance' columns of the dataset to predict the relevance of certain advertisements.

## Preprocessing

I first removed the word 'ad-' from the advertisement columns for easier modeling and analysis and converted the columns
to integers. I also remnamed the 'relevance' and 'ad' columns from 'c0' and 'c1'.

## Feature Engineering

I encoded the categorical columns into numerical format with StringIndexer and OneHotEncoder to prepare them to be
used in the machine learning model.

## Training

I fit the categorical columns into a 'features' vector utilizing VectorAssembler while the relevance column is
converted to an integer 'label' column as the target variable.

## Output

I then created a model to display the outcome of our linear regression model.
![image](https://github.com/simolevy/DataAnalysisPortfolio/assets/97460770/2ae93689-38db-45f7-b537-a01fde03887b)

This will allow us to better assess the performance of our model.

## Validation

I also utilized a confusion matrix to visualize the model's performance and view any false negatives and positives.
![image](https://github.com/simolevy/DataAnalysisPortfolio/assets/97460770/3ff2ae4b-bcb9-4601-8b10-7727be46eeba)


## Results

Based on the results of our model and confusion matrix, the logistic regression model was effective in predicting the relevance
of the ads. This is likely due to the quality of the data from our dataset and its relative simplicity.

