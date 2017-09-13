import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

#import modelling libraries
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
import scipy.stats as scs
from statsmodels.stats.outliers_influence import variance_inflation_factor

#plotting
from statsmodels.graphics.regressionplots import *

class realEstateModel(object):

    def load_data(self, filepath, dates=None, yr_first=False):
        '''
        Load data from csv file located in the data folder
        '''
        if dates:
            df = pd.read_csv(filepath, parse_dates=dates, date_parser = pd.tseries.tools.to_datetime)
        else:
            df = pd.read_csv(filepath)
        return df

    def prepare_data(self, filepath, outcome, dates=None, yr_first=False):
        '''
        Engineer new features and return in format for modeling and visualization
        '''
        df = self.load_data(filepath)
        #make new variables
        df['ziplarge'] = df['zipcode'].apply(lambda x: str(x)[:-1])
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
        df['age'] = df['date'].apply(lambda x: x.year)-df['yr_built']
        df['renovated'] = df['yr_renovated'].apply(lambda x: 1 if x>0 else 0)
        df['ln_price'] = df['price'].apply(lambda x: np.log(x))
        df['grade_sm'] = df['grade'].replace({3:1, 4:1, 5:1, 6:1})
        # df['binned_age'] = pd.cut(df.age, bins = 7)
        df['sqrt_living'] = df['sqft_living'].apply(lambda x: np.sqrt(x))
        df = df[df.bedrooms<15]

        # self.numericvars = ['ln_price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'age']
        self.allnumericvars = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'age', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
        self.numericvars = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'age']
        self.categoricvars = ['ziplarge', 'floors', 'waterfront', 'view', 'condition', 'grade', 'renovated']
        #what to do with 'lat', 'long'?
        self.df = df.drop(['zipcode', 'date', 'lat', 'long', 'yr_built', 'yr_renovated'], axis=1)
        self.df.drop_duplicates(subset='id', keep='last', inplace=True)
        self.X = pd.get_dummies(self.df, columns=['ziplarge', 'bedrooms', 'condition', 'grade', 'view', 'floors'], drop_first=True).drop(['id', 'price', 'ln_price'], axis=1)
        # self.y = self.df['ln_price']
        self.outcome = outcome
        self.y = self.df[self.outcome]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.cols = self.X.columns
        #scale features
        self.X_data = scale(self.X_train)
        #do viz as well
        # self.scatterplots()
        # self.boxplots()
        self.ln_boxplots()
        # self.histograms()
        self.correlation()
        return self.df

    def histograms(self):
        '''
        VISUALIZE distributions of numeric variables
        '''
        for var in self.numericvars:
            plt.hist(var, data=self.df, bins=20)
            plt.savefig('figures/realestate_{}_histogram.pdf'.format(var))
            plt.close()

    def correlation(self):
        '''
        corrlelation matrix of all numeric vars
        '''
        corr = self.df[self.allnumericvars].corr()
        plt.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns);
        plt.yticks(range(len(corr.columns)), corr.columns);
        print(self.df[self.allnumericvars].corr())
        # plt.tight_layout()
        plt.savefig('figures/realestate_correlationmatrix.pdf')
        plt.close()

    def scatterplots(self):
        '''
        VISUALIZE relationship between price and numeric variables
        '''
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax = pd.tools.plotting.scatter_matrix(self.df[self.numericvars], ax=ax, diagonal='kde')
        plt.savefig('figures/realestate_price_numericScatterplots.pdf')
        plt.close()

    def ln_boxplots(self):
        '''
        VISUALIZE relationship between price and categical variables
        '''
        for var in self.categoricvars:
            sns.boxplot(y='ln_price', x=var, data=self.df)
            plt.savefig('figures/realestate_lnprice_{}_Boxplot.pdf'.format(var))
            plt.close()

    def boxplots(self):
        '''
        VISUALIZE relationship between price and categical variables
        '''
        for var in self.categoricvars:
            sns.boxplot(y='price', x=var, data=self.df)
            plt.savefig('figures/realestate_price_{}_Boxplot.pdf'.format(var))
            plt.close()

    def ols_model(self, df, formula, modelType):
        '''
        Model variables specified
        '''
        model = ols(formula = formula, data=df)
        # model = sm.mixedlm(formula=formula, data=df, groups=df["id"])
        self.res = model.fit()
        print("Model: {}".format(formula))
        print(self.res.summary())
        self.modelType = modelType
        # self.ols_model_viz(df)
        return self.res


    def ols_model_viz(self, df):
        '''
        Visulize residuals and diagnostics
        '''
        plt.scatter(self.res.fittedvalues, df[self.outcome])
        plt.xlabel('Fitted Values')
        plt.ylabel(self.outcome)
        plt.title('Fitted Values Plot')
        plt.savefig('figures/model{}FittedValues.pdf'.format(self.modelType))
        plt.close()
        plt.scatter(self.res.fittedvalues, self.res.outlier_test()['student_resid'])
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.savefig('figures/model{}_residuals.pdf'.format(self.modelType))
        plt.close()
        fig = plt.figure(figsize=(20,12))
        fig = sm.graphics.plot_partregress_grid(self.res, fig=fig)
        plt.savefig('figures/model{}_partialRegression.pdf'.format(self.modelType))
        plt.close()
        influence = self.res.get_influence()
        (c, p) = influence.cooks_distance
        plt.stem(np.arange(len(c)), c, markerfmt=",")
        plt.xlabel('Observations')
        plt.ylabel('Cooks Distance')
        plt.title('Cooks Distance plot')
        plt.savefig('figures/model{}_CooksDistance.pdf'.format(self.modelType))
        plt.close()
        plot_leverage_resid2(self.res.fittedvalues)
        plt.savefig('figures/model{}_LeverageResid.pdf'.format(self.modelType))
        plt.close()
        influence_plot(self.res.fittedvalues)
        plt.savefig('figures/model{}_InfluencePlot.pdf'.format(self.modelType))
        plt.close()

    def ols_regular(self, df, formula, modelType):
        '''
        Model variables specified
        '''
        model = ols(formula = formula, data=df)
        # model = sm.mixedlm(formula=formula, data=df, groups=df["id"])
        self.res = model.fit()
        print("Model: {}".format(formula))
        print(self.res.summary())
        self.modelType = modelType
        return self.res

    def rmse(self, theta, thetahat):
        '''
        Compute Root-mean-squared-error
        '''
        return np.sqrt(np.mean((theta - thetahat) ** 2))

    def calc_linear(self):
        '''
        Fit model on training set and predict on test set, then return RMSE
        '''
        linear = LinearRegression()
        linear.fit(self.X_train, self.y_train)

        train_predicted = linear.predict(self.X_train)
        test_predicted = linear.predict(self.X_test)

        trainErr = self.rmse(self.y_train, train_predicted)
        valErr = self.rmse(self.y_test, test_predicted)

        return (trainErr, valErr)

    def lasso(self, alphas):
        '''
        Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
        '''
        # Create an empty data frame
        df = pd.DataFrame()
        # Create a column of feature names
        df['Feature Name'] = self.cols
        # For each alpha value in the list of alpha values,
        for alpha in alphas:
            # Create a lasso regression with that alpha value,
            lasso = Lasso(alpha=alpha, max_iter=10000)
            # Fit the lasso regression
            lasso.fit(self.X_data, self.y_train)
            # Create a column name for that alpha value
            column_name = 'Alpha = %f' % alpha
            # Create a column of coefficient values
            df[column_name] = lasso.coef_
        # Return the datafram
        return df

    def mse_lasso(self, alpha=0.5):
        lasso = Lasso(alpha=alpha)
        fit = lasso.fit(self.X_data, self.y_train)
        y_pred = lasso.predict(self.X_test)
        return mean_squared_error(self.y_test, y_pred)

    def test_alphas(self, model, alphas):
        '''
        INPUT
            model: the name of a model class (eg Ridge)
        OUTPUT
            ARRAYS k_fold_train_error, k_fold_test_error:
                returns the mean error of 10 fold cross validation for both train and test set, for all alphas
        '''
        k_fold_train_error = np.zeros(len(alphas))
        k_fold_test_error = np.zeros(len(alphas))
        for i, a in enumerate(alphas):
            regmodel = model(alpha=a, normalize=True)
            kf = KFold(self.X_train.shape[0], n_folds=10)
            train_error, test_error = self.KFoldCVModel(regmodel, kf)
            k_fold_train_error[i] = np.mean(train_error)
            k_fold_test_error[i] = np.mean(test_error)
        return k_fold_train_error, k_fold_test_error

    def KFoldCVModel(self, model, kf, n_folds=10):
        '''
        INPUT
            model: the name of a model class (eg Ridge)
            kf: indices for splitting train/test data for each fold
            n_folds: number of folds
        OUTPUT
            ARRAYs train_error, test_error: array of error for each fold
        '''
        for i, (train, validation) in enumerate(kf):
            # model.fit(self.X_train.values[train], self.y_train[train])
            model.fit(self.X_train.values[train], self.y_train.values[train])
            train_error = np.empty(n_folds)
            val_error = np.empty(n_folds)
            train_error[i] = mean_squared_error(self.y_train.values[train], model.predict(self.X_train.values[train]))
            val_error[i] = mean_squared_error(self.y_train.values[validation], model.predict(self.X_train.values[validation]))
        return train_error, val_error

    def plot_lasso(self, alphas):
        '''
        function to plot mean squared error of lasso regression over a range of alphas
        SHOWS plot
        RETURNS alpha with lowest mse
        '''
        n_folds = 10
        params = np.zeros((len(alphas), self.X.shape[1]))

        for i, a in enumerate(alphas):
            lasso = Lasso(alpha=a).fit(self.X_data, self.y_train)
            params[i] = lasso.coef_
        #
        k_fold_train_error, k_fold_test_error = self.test_alphas(Lasso, alphas)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(alphas, params)
        ax.set(xlabel='Alpha', ylabel='Coefficients', title='Alphas vs. Beta Coefficients')

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(alphas, k_fold_train_error)
        ax2.plot(alphas, k_fold_test_error)
        ax2.vlines(alphas[k_fold_test_error.argmin()], 0, k_fold_test_error.max())
        ax2.set(xlabel='Alpha', ylabel='MSE', title='Alphas vs. MSE')
        ax2.set_xscale('log')
        plt.savefig("figures/realestate_lasso_plot.pdf")
        plt.close()
        return alphas[k_fold_test_error.argmin()]

    def vifs(self, x):
    	'''
    	Input x as a DataFrame, calculates the vifs for each variable in the DataFrame.
    	DataFrame should not have response variable.
    	Returns dictionary where key is column name and value is the vif for that column.
    	Requires scipy.stats be imported as scs
    	'''
    	vifs = []
    	for index in range(x.shape[1]):
    		vifs.append(round(variance_inflation_factor(x.values, index),2))
    	return vifs

if __name__ == "__main__":

    rmod = realEstateModel()
    # fit model with price vs sqft_living
    # df_price = rmod.prepare_data('kc_house_data.csv', 'price')
    # rmod.ols_model(df_price, "price~sqft_living", "PriceSimple")
    # fit model wRT sqft_living
    df = rmod.prepare_data('kc_house_data.csv', 'ln_price')
    # rmod.ols_model(df, "ln_price~sqft_living", "Simple")
    # #fit model wRT sqft_living, grade, view, waterfront
    rmod.ols_model(df, "ln_price~ age + renovated + sqft_living + C(grade) + C(view) + waterfront", "Full")
    # #fit model wRT sqft_living, grade, view, waterfront, zipcode simplified
    # rmod.ols_model(df, "ln_price~ age + renovated + sqft_living + C(grade) + C(view) + C(ziplarge) + waterfront", "Complex")
    # #look at VIF for all features
    # vifs = rmod.vifs(df.drop(['id', 'price', 'ln_price', 'ziplarge', 'sqrt_living'], axis=1))
    #
    # print(rmod.calc_linear())
    # alphamin = rmod.plot_lasso(np.linspace(0.0005, 1, 100))
    # print(rmod.lasso([0.005, 0.01, 0.05]))
    #
    # rmod.ols_model(df, "ln_price~ bathrooms + age + renovated + sqft_living + C(grade) + C(view) + C(ziplarge) + waterfront", "Final")
