import pandas as pd
from faker import Faker
import numpy as np

def generate_data(n):
    fake = Faker()
    Faker.seed(42)

    data = {
        "card_number": [str(fake.random_number(digits=16, fix_len=True)) for _ in range(n)],
        "Phone 1": [str(fake.random_number(digits=10, fix_len=True)) for _ in range(n)],
        "Phone 2": [str(fake.random_number(digits=10, fix_len=True)) for _ in range(n)],
        "Email": [str(fake.random_number(digits=10, fix_len=True)) for _ in range(n)],
        "IP": ["".join(str(fake.random_int(min=0, max=255)).zfill(3) for _ in range(4)) for _ in range(n)],
    }

    df = pd.DataFrame(data)

    # 35% of records with same Email value but different other column values
    email1 = str(fake.random_number(digits=10, fix_len=True))
    for i in range(int(n * 0.35)):
        df.at[i, 'Email'] = email1

    # Next 35% of records with another same Email value but different other column values
    email2 = str(fake.random_number(digits=10, fix_len=True))
    for i in range(int(n * 0.35), int(n * 0.7)):
        df.at[i, 'Email'] = email2

    return df

n = 10000
df = generate_data(n)
df.to_csv('SourceData_For_OutlierDataset.csv', index=False)

# Read the CSV file back into DataFrame to ensure all values are strings
df_actual = pd.read_csv('SourceData_For_OutlierDataset.csv', dtype=str)

# Create a DataFrame with NaNs for all duplicate email rows
df_with_nan_emails = df_actual.copy()

# Identify all duplicate email values
duplicate_emails = df_actual['Email'][df_actual.duplicated(subset=['Email'], keep=False)].unique()

# Set the 'Email' column to NaN for all rows with duplicate email values
df_with_nan_emails.loc[df_with_nan_emails['Email'].isin(duplicate_emails), 'Email'] = np.nan

# Save the DataFrame with NaNs to CSV
df_with_nan_emails.to_csv('OutlierDataset.csv', index=False)

print("Data generation complete. Saved to 'generated_data.csv' and 'outlier_subset_with_nan_emails.csv'.")
