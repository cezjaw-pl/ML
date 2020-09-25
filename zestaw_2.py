# importing libraries and setting options
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skll
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# opening csv file
df = pd.read_csv("J:/Studia/Uczenie_maszynowe/survey_results_public.csv", usecols=["Respondent",
                 "YearsCodePro", "Age1stCode"], index_col="Respondent")

# replacing string values and converting to integers
df['YearsCodePro'].replace(to_replace=['Less than 1 year', 'More than 50 years'], value=['0.5', '51'], inplace=True)
df['Age1stCode'].replace(to_replace=['Younger than 5 years', 'Older than 85'], value=['5', '85'], inplace=True)
df.dropna(inplace=True)
df['YearsCodePro'] = df['YearsCodePro'].astype(float)
df['Age1stCode'] = df['Age1stCode'].astype(int)

# depicting the correlation and plotting the boxplot
print(df.corr(), df.describe())
plt.boxplot('YearsCodePro', data=df)
plt.show()

# calculating mean and standard deviation, nextly a Z-score
df_mean = df.mean()
df_sd = df.std()
z_score = (df-df_mean)/df_sd
# removing the outliers from DataFrame basing on Z-score
df_wo_outlines = df[~((z_score > 3) | (z_score < -3)).any(axis=1)]

# depicting the correlation and plotting the boxplot after removing outlines
print(df_wo_outlines.corr(), df_wo_outlines.describe())
plt.boxplot('YearsCodePro', data=df_wo_outlines)
plt.show()

# reshaping the independent value to get 2-dimensional array
X = df_wo_outlines['Age1stCode'].values.reshape((-1, 1))
# creating linear regression object, fitting the model and determaining the equation with intercept and coefficient
reg = skll.LinearRegression().fit(X, df_wo_outlines['YearsCodePro'])
print(reg.intercept_)
print(reg.coef_)
# calculating the mean square error
mse = mean_squared_error(df_wo_outlines[['YearsCodePro']], reg.predict(X))
print('MSE: %.2f' % mse)

# plotting the regression model
sns.regplot(y='Age1stCode', x='YearsCodePro', scatter_kws={"color": "blue"}, line_kws={"color": "red"},
            x_estimator=np.mean, data=df_wo_outlines, logx=True)
plt.show()
