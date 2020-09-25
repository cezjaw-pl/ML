# importing pandas and sklearn libraries and setting options
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklp
import sklearn.linear_model as skll
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# opening csv file
df = pd.read_csv("J:/Studia/Uczenie_maszynowe/survey_results_public.csv", usecols=["Respondent", "OrgSize", "YearsCode",
                     "YearsCodePro", "LastHireDate","ConvertedComp","Student","UndergradMajor"], index_col="Respondent")

# replacing string values and converting to floats
df['YearsCodePro'].replace(to_replace=['Less than 1 year','More than 50 years'], value=['0', '51'], inplace=True)
df['YearsCode'].replace(to_replace=['Less than 1 year','More than 50 years'], value=['0', '51'], inplace=True)

df['LastHireDate'].replace(to_replace=["I've never had a job",
'NA - I am an independent contractor or self employed', 'Less than a year ago', '1-2 years ago', '2-3 years ago',
'3-4 years ago', 'More than 4 years ago'], value=['0', '0', '0.5', '1', '2', '3', '4'], inplace=True)

df['OrgSize'].replace(to_replace=['Just me - I am a freelancer, sole proprietor, etc.', '2-9 employees',
'10 to 19 employees', '20 to 99 employees', '100 to 499 employees', '500 to 999 employees', '1,000 to 4,999 employees',
'5,000 to 9,999 employees', '10,000 or more employees'],
value=['1', '5', '15', '60', '300', '750', '3000', '7500', '10000'], inplace=True)
df.dropna(inplace=True)
df['OrgSize'] = df['OrgSize'].astype(float)
df['YearsCode'] = df['YearsCode'].astype(float)
df['YearsCodePro'] = df['YearsCodePro'].astype(float)
df['LastHireDate'] = df['LastHireDate'].astype(float)

# replacing string values for column Student
df['Student'].replace(to_replace=['No', 'Yes, full-time', 'Yes, part-time'], value=['0', '1', '1'], inplace=True)
df['Student'] = df['Student'].astype(int)

# creating a binarize-labeled numpy arraies with usage of column UndergradMajor
label_bin = sklp.LabelBinarizer()
transformed_label_bin = label_bin.fit_transform(df['UndergradMajor'])
# transforming into DataFrame
df_label_bin = pd.DataFrame(transformed_label_bin, columns=label_bin.classes_, index=df.index)
# joining to primary DataFrame
df = df.join(df_label_bin)
df['UndergradMajor'] = df_label_bin

# subsetting new DataFrame with float-typed columns
df2 = pd.DataFrame(df[['OrgSize', 'YearsCode', "YearsCodePro", "LastHireDate","ConvertedComp"]])
# dropping NA values
df2.dropna(inplace=True)

# calculating 1st, 3rd quantile and IQR
Q1 = df2.quantile(0.25)
Q3 = df2.quantile(0.75)
IQR = Q3 - Q1
#removing the outliers from DataFrame basing on quantiles
df2_out_quantiles = df2[~((df2 < (Q1 - 1.5 * IQR)) |(df2 > (Q3 + 1.5 * IQR))).any(axis=1)]

#calculating mean and standard deviation, nextly a Z-score
df2_mean = df2.mean()
df2_sd = df2.std()
z_score=(df2-df2_mean)/df2_sd
#removing the utliers from DataFrame basing on Z-score
df2_out_sd = df2[~((z_score > 3) | (z_score < -3)).any(axis=1)]

# seeking for correlations
df.corr()
# setting variables
x1 = df2_out_quantiles['OrgSize']
x2 = df2_out_quantiles['YearsCodePro']
y = df2_out_quantiles['ConvertedComp']

# reshaping the independent value to get 2-dimensional array
X_s = x2.values.reshape((-1, 1))
# dividing observations into training and testing parts
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_s, y, test_size=0.01, random_state=0)

# creating linear regression object
regressor_s = skll.LinearRegression()
# fitting the training sets and determaining the equation with intercept and coefficient
regressor_s.fit(X_s_train, y_s_train)
print(regressor_s.intercept_)
print(regressor_s.coef_)
# using the predict function to get predicted value
y_s_pred = regressor_s.predict(X_s_test) # regressor.intercept_ + regressor.coef_ * x2
# calculating the mean square error
mse_s = mean_squared_error(y_s_test, y_s_pred)
print('MSE: %.2f'%mse_s)

# plotting the regression model
plt.scatter(X_s_train, y_s_train,  color='black')
plt.plot(X_s_train, regressor_s.predict(X_s_train), color='blue', linewidth=3)
plt.title("YearsCodePro vs ConvertedComp (Training set)")
plt.xlabel("YearsCodePro")
plt.ylabel("ConvertedComp")
plt.show()
plt.scatter(X_s_test, y_s_test,  color='black')
plt.plot(X_s_train, regressor_s.predict(X_s_train), color='blue', linewidth=3)
plt.title("YearsCodePro vs ConvertedComp (Testing set)")
plt.xlabel("YearsCodePro")
plt.ylabel("ConvertedComp")
plt.show()

# combining two independent values
X_m = pd.concat([x1, x2], axis=1)
# creating linear regression object, fitting the model and determaining the equation with intercept and coefficient
regressor_m = skll.LinearRegression().fit(X_m, y)
print(regressor_m.intercept_)
print(regressor_m.coef_)
# using the predict function to get predicted value
y_m_pred = regressor_m.predict(X_m)
# calculating the mean square error
mse_m = mean_squared_error(y, y_m_pred)
print('MSE: %.2f'%mse_m)

# plotting the regression model
sns.regplot(x=X_m['OrgSize'], y=y, scatter_kws={"color": "blue"}, line_kws={"color": "red"}, x_estimator=np.mean)
plt.show()

# creating linear regression object, fitting the model and determaining the equation with intercept and coefficient
regressor_c = skll.LinearRegression().fit(df[['OrgSize', 'YearsCodePro', 'Student', 'UndergradMajor']], df[['ConvertedComp']])
print(regressor_c.intercept_)
print(regressor_c.coef_)
# calculating the mean square error
mse_c = mean_squared_error(df[['ConvertedComp']], regressor_c.predict(df[['OrgSize', 'YearsCodePro', 'Student', 'UndergradMajor']]))
print('MSE: %.2f'%mse_c)
# plotting the regression model
sns.lmplot(x='YearsCodePro', y='ConvertedComp', hue="Student", data=df)
plt.show()
