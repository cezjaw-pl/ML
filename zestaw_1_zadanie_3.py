# importing pandas and setting options
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# reading the csv file and naming columns
df_train = pd.read_csv("J:/Studia/Uczenie_maszynowe/first-assignment-master/train.tsv", header=None, sep='\t')
df_description = pd.read_csv("J:/Studia/Uczenie_maszynowe/first-assignment-master/description.csv")
df_train = df_train.rename(columns={0: 'Price', 1: 'Rooms', 2: 'Area', 3: 'Stage', 4: 'Location', 5: 'Description'})
df_description.columns = ['Stage', 'Stage description']

# adding values to missing stage descriptions
values = {'Stage': [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'Stage description': ['pierwsze piętro', 'czwarte piętro',
'piąte piętro', 'szóste piętro', 'siódme piętro', 'ósme piętro', 'dziewiąte piętro', 'dziesiąte piętro', 'jedenaste piętro',
'dwunaste piętro', 'trzynaste piętro', 'czternaste piętro', 'piętnaste piętro', 'szesnaste piętro']}
added_values = pd.DataFrame(values)

# appending original Data Frame with new values
df_description = df_description.append(added_values, ignore_index=True)
# merging data frames receiving an extra column with stage description
df_merged = pd.merge(df_train, df_description, on='Stage', how='left')
df_merged.to_csv("out2.csv", float_format='%.2f', index=False)
