import pandas as pd
from faker import Faker
import numpy as np

def genrate_data(n):
    fake = Faker()
    data = {
        "card_number" : [fake.random_number(digits=16, fix_len=True) for _ in range(n)],
        "Phone 1": [fake.random_number(digits=10, fix_len=True) for _ in range(n)],
        "Phone 2": [fake.random_number(digits=10, fix_len=True) for _ in range(n)],
        "Email": [fake.random_number(digits=10, fix_len=True) for _ in range(n)],
        "IP": ["".join(str(fake.random_int(min=0 , max=255)) for _ in range(4)) for _ in range(n)],
    }
    return pd.DataFrame(data)

n=100

df= genrate_data(n)
df.to_csv('SourceDataSet_AllRecord_FromSource.csv', index=False)

df_subset = df.copy()

# 65% records same as original
df_subset.iloc[int(n*0.65):] = df.iloc[int(n*0.65):]

# 5% records completely missing
df_subset.iloc[int(n*0.65):int(n*0.70)] = np.nan

# Drop rows with all nan values
df_subset = df_subset.dropna(how='all')

# 5% records missing card_number
df_subset.loc[int(n*0.70):int(n*0.75), 'card_number'] = np.nan

# 5% records missing Phone 1
df_subset.loc[int(n*0.75):int(n*0.80), 'Phone 1'] = np.nan

# 5% records missing Phone 2
df_subset.loc[int(n*0.80):int(n*0.85), 'Phone 2'] = np.nan

# 5% records missing Email
df_subset.loc[int(n*0.85):int(n*0.90), 'Email'] = np.nan

# 5% records missing IP
df_subset.loc[int(n*0.90):int(n*0.95), 'IP'] = np.nan


#creat list of numerical columns
numeric_columns = ["card_number", "Phone 1", "Phone 2", "Email", "IP"]

#Apply Lambda Function only to numeric column

for column in numeric_columns:
    df_subset[column] = df_subset[column].apply(lambda x: str(int(x)) if pd.notnull(x) else '')

df_subset.to_csv('MissingRecord_MissingColumnData.csv',index=False)

# Read the CSV files back into DataFrames
df_actual = pd.read_csv('SourceDataSet_AllRecord_FromSource.csv')
df_subset = pd.read_csv('MissingRecord_MissingColumnData.csv')

# Convert specified columns to string to avoid scientific notation
for column in numeric_columns:
    df_actual[column] = df_actual[column].astype(str)
    df_subset[column] = df_subset[column].astype(str)

# Merge DataFrames to find non-matching records
merged_df = df_subset.merge(df_actual, on=["card_number", "Phone 1", "Phone 2", "Email", "IP"], how='left', indicator=True)

# Filter non-matching records
non_matching_records = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

# Print all columns and rows in the DataFrame without truncation
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.expand_frame_repr', False)  # Prevent line-wrapping


print(df.iloc[64:90])