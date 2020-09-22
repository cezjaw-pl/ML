# importing pandas and matplotlib libraries and setting options
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# reading the csv file and naming columns
df = pd.read_csv("J:/Studia/Uczenie_maszynowe/survey_results_public.csv",
    usecols=["Respondent", "Gender", "YearsCodePro", "Age"], index_col="Respondent")
# replacing string values into integers
df.replace(to_replace='Less than 1 year', value='0', inplace=True)
df.replace(to_replace='More than 50 years', value='51', inplace=True)
df.dropna(inplace=True)
df['YearsCodePro'] = df['YearsCodePro'].astype(int)

# calculating mean and standard deviation, nextly a Z-score
df_mean = df.mean()
df_sd = df.std()
z_score = (df-df_mean)/df_sd
# removing the outliers from DataFrame basing on Z-score
df_out_sd = df[~((z_score > 3) | (z_score < -3)).any(axis=1)]

# creating a plot from full DataFrame
df_out_sd.plot(kind="box", x="Age", y="YearsCodePro", color="black")

# subsetting the DataFrame depending on column Gender
df_men = df_out_sd.loc[(df_out_sd['Gender'] == "Man")]
df_women = df_out_sd.loc[(df_out_sd['Gender'] == "Woman")]
# ploting DataFrames
df_women.plot(kind="scatter", x="Age", y="YearsCodePro", color="green")
df_men.plot(kind="hist", x="Age", y="YearsCodePro", color="blue")
plt.show()
