# Cruise Ship Crew Count Prediction Model

Dataset --> https://www.kaggle.com/datasets/shauryajain/cruise-ship-data

In this project I used the dataset features of <i>"Tonnage", "passengers", "length", "cabins", and "passenger_density" </i> to predict the
number of crew numbers that will be needed on a cruise ship.

## Preprocessing

I altered the dataset by changing the numerical values from String variables into Double or Integer values for easier computations.
I also changed some variables into categorical variables for easier plotting and manipulation.
I created a new row of "features" containing the variables tonnage, passengers, length, cabins, and passenger_density, with a second 
row titled crew_member to prepare for training and testing. 

## Training

I used a 70:30 training and test random split to create my linear regression model.
I then trained and fit the model using the LinearRegression model imported from PySpark. 

## Output

I created a method to display summary statistics of the linear regression model:

```
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
```

## Cross Validation

I then cross validated by viewing the testing data against the training data.


<img width="274" alt="image" src="https://github.com/simolevy/CruiseCrewPrediction/assets/97460770/e88d94d3-9b92-4835-9014-50badd23ac20">


## Results

My model summary output stated that the R^2 value is 0.935800, which means that the model explains around 94% of the variance.
This is very high and shows that our model did a very accurate job of predicting the number of crew members needed on a cruise ship 
based on the training features that we chose!
