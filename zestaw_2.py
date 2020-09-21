#importing csv, pandas and sklearn libraries and setting options
import pandas as pd
import sklearn.preprocessing as skl
pd.set_option('display.width',None)
pd.set_option('display.max_colwidth', None)
df = pd.read_csv("J:/Studia/Uczenie_maszynowe/survey_results_public.csv",usecols=["Respondent", "OrgSize", "YearsCode", "YearsCodePro", "LastHireDate","ConvertedComp","Student","UndergradMajor"], index_col="Respondent")
#replacing string values and converting to floats
df.replace(to_replace='Less than 1 year', value='0', inplace=True)
df.replace(to_replace='More than 50 years', value='51', inplace=True)
df.replace(to_replace="I've never had a job", value='0', inplace=True)
df.replace(to_replace="NA - I am an independent contractor or self employed", value='0', inplace=True)
df.replace(to_replace='Less than a year ago', value='0.5', inplace=True)
df.replace(to_replace='1-2 years ago', value='1', inplace=True)
df.replace(to_replace='2-3 years ago', value='2', inplace=True)
df.replace(to_replace='3-4 years ago', value='3', inplace=True)
df.replace(to_replace='More than 4 years ago', value='4', inplace=True)
df.replace(to_replace='Just me - I am a freelancer, sole proprietor, etc.', value='1', inplace=True)
df.replace(to_replace='2-9 employees', value='2', inplace=True)
df.replace(to_replace='10 to 19 employees', value='10', inplace=True)
df.replace(to_replace='20 to 99 employees', value='20', inplace=True)
df.replace(to_replace='100 to 499 employees', value='100', inplace=True)
df.replace(to_replace='500 to 999 employees', value='500', inplace=True)
df.replace(to_replace='1,000 to 4,999 employees', value='1000', inplace=True)
df.replace(to_replace='5,000 to 9,999 employees', value='5000', inplace=True)
df.replace(to_replace='10,000 or more employees', value='10000', inplace=True)
df.replace(to_replace='No', value='0', inplace=True)
df.replace(to_replace='Yes, full-time', value='1', inplace=True)
df.replace(to_replace='Yes, part-time', value='1', inplace=True)
df.dropna(inplace=True)
df['OrgSize'] = df['OrgSize'].astype(float)
df['YearsCode'] = df['YearsCode'].astype(float)
df['YearsCodePro'] = df['YearsCodePro'].astype(float)
df['LastHireDate'] = df['LastHireDate'].astype(float)
df['Student'] = df['Student'].astype(int)
#seeking for correlations
df.corr()
#setting variables
x1=df['OrgSize']
x2=df['YearsCodePro']
y=df['ConvertedComp']
#creating a binarize lebeled numpy arraies with usage of column UndergradMajor
label_bin = skl.LabelBinarizer()
transformed_label_bin = label_bin.fit_transform(df['UndergradMajor'])
#tranfosrming into DataFrame
df_label_bin= pd.DataFrame(transformed_label_bin, columns=label_bin.classes_)
#joining to primary DataFrame
df = df.join(df_label_bin)
