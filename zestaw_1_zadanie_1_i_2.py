# importing pandas library and setting options
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# reading the csv file and naming columns
df = pd.read_csv("J:/Studia/Uczenie_maszynowe/first-assignment-master/train.tsv", header=None, sep='\t')
df = df.rename(columns={0: 'Price', 1: 'Rooms', 2: 'Area', 3: 'Stage', 4: 'Location', 5: 'Description'}, inplace=False)
# Column Price was multiplicated by 1000 to get the value in units, not in thousands
df['Price'] = df['Price']*1000

# 1)
# looping through the data frame to sum the prices.
sum = 0
for index, row in df.iterrows():
    sum += (row['Price'])
# calculating an average price of the flats
avg_price = sum/len(df)
# converting the average price to data frame and exporting to csv file
avg_price = pd.DataFrame(round(avg_price), columns=['Average'], index=['a'])
avg_price.to_csv("out0.csv", header=False, index=False)

# 2)
# creating new data frame with added column - Price per square meters
df2 = df.assign(Price_per_m2=round(df['Price']/df['Area']))
# looping through the data frame to sum the prices per square meters
sum2 = 0
for index, row in df2.iterrows():
    sum2 += (row['Price_per_m2'])
# calculating an average price per square meter
avg_price_per_m2 = sum2/len(df2)
# subsetting the data frame to get flats with at least 3 rooms
# and price per square meter lower than average price per square meter
df3 = df2.loc[(df2['Rooms'] >= 3) & (df2['Price_per_m2'] < avg_price_per_m2)]
# exporting subsetted data frame to csv file
df3.to_csv("out1.csv", columns=['Rooms', 'Price', 'Price_per_m2'], header=False, float_format='%.2f', index=False)
