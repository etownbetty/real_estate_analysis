# Real Estate Analysis

A number of variables were available and potentially related to the price of houses in the SF bay area, as well as the variable to be used as the outcome variable = price (price at last sale).
A quick look shows that there were variables related to house and lot size (sqft_living, sqft_above, sqft_basement, sqft_lot, sqft_living15, sqft_lot15), location, whether or not there was a view or was on the waterfront, whether the house was renovated, and some general condition information. All of these could be related to the house value, but we can potentially find objectively if the variables are important is determining the house sale price and how important.
I engineered some variables that could also be predictive of house price, like age of the house, and whether it was renovated (as opposed to when it was renovated), and I collapsed the zip code variable into larger bins, based on the first 4 numbers of the zip code which will group houses in nearby zipcodes.

## Exploratory

Before fitting a model, I wanted to see how the variables were distributed and if there were any apparent relationships between them and the outcome price variable. I plotted histograms of the continuous variables, a scatterplot matrix of the continuous variables and box plots of the categorical variables with respect to the sale price of the houses.

## A simple model

As a starting point, I fit a model of price vs sqft, which had an rSquared value of 0.492, and showed that sqft was a significant predictor or house price, but upon inspection of the residual plots, there is a pattern to the residuals - the larger the house price, the larger the errors, seen below.
<center>
  <img src="./figures/modelPriceSimple_residuals.pdf" alt="Simple Model Residuals">
</center>

So I took a log transform of the house price and refit the model. The sqft feature stayed significant, and the residuals were more randomly distributed around a mean of zero, with the exception of some outliers, which were also seen in the original residual plot.
<center>
  <img src="./figures/modelSimple_residuals.pdf" alt="Simple Model Log-transformed Residuals">
</center>

## A full model

I then added some additional features that appeared to have a relatively linear relationship with the log price variable, according to the correlation plot matrix and box plots against the log price outcome, and didn't have any redundancy with each other. The features that I added, age of house, whether it had been renovated, the grade, the view, and waterfront property status were all significantly associated with the log price outcome and more of the variance was explained in the model - the adjusted R-Square measure went up to 0.638.
The residual plot showed that the errors were randomly distributed around a mean of zero and the outliers appear to have been explained better with a more complex model.

[[https://github.com/etownbetty/real_estate_analysis/figures/modelFull_residuals.pdf]]
<center>
  <img src="./figures/modelFull_residuals.pdf" alt="Full Model Log-transformed Residuals">
</center>

## A complex model

I also wanted to add some information about the location of the houses with respect to price, so I added a categorical simplified zip code variable into my model - this further improved the amount of variance explained with an adjusted R-Squared value of 0.691.

## Lasso for the win

I cross-checked my model using a lasso procedure to do feature selection from all the available features. Using an alpha=0.01, I found that my complex model had all significant features that didn't have any redundancy to them in it. I also added a bathroom feature, which was the number of bathrooms, this further increased the adjusted R-Squared value to 0.696.

My final model to predict housing prices is:

ln_price = bathrooms + age + renovated + sqft_living + grade + view + ziplarge + waterfront

Where ziplarge is my feature with super-zip codes, i.e. 4 number zip codes
