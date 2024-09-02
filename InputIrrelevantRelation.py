import pandas as pd
from faker import Faker
import numpy as np
import random

def generate_data(n, num_cards=5):
    fake = Faker()
    Faker.seed(42)

    data = {
        "card_number": [],
        "Phone 1": [],
        "Phone 2": [],
        "Email": [],
        "IP": []
    }

    for _ in range(num_cards):
        # Common values for 90% records
        card_number_common = str(fake.random_number(digits=16, fix_len=True))
        phone1_common = str(fake.random_number(digits=10, fix_len=True))
        phone2_common = str(fake.random_number(digits=10, fix_len=True))
        email_common = str(fake.random_number(digits=10, fix_len=True)).zfill(10)  # Email with 10 digits

        # Generate a common numeric IP address
        ip_common = ''.join(str(fake.random_int(min=0, max=255)).zfill(3) for _ in range(4))

        email_less_frequency = str(fake.random_number(digits=10, fix_len=True)).zfill(10)  # Email with 10 digits

        # Generate a different numeric IP address
        ip_less_frequency = ''.join(str(fake.random_int(min=0, max=255)).zfill(3) for _ in range(4))

        # 90% of records with the same card_number, email, Phone 1, Phone 2, IP
        for _ in range(int(n * 0.9 / num_cards)):
            data["card_number"].append(card_number_common)
            data["Phone 1"].append(phone1_common)
            data["Phone 2"].append(phone2_common)
            data["Email"].append(email_common)
            data["IP"].append(ip_common)

        # 10% of records with the same card_number, Phone 1, Phone 2, but different email and different IP
        for _ in range(int(n * 0.1 / num_cards)):
            data["card_number"].append(card_number_common)
            data["Phone 1"].append(phone1_common)
            data["Phone 2"].append(phone2_common)
            data["Email"].append(email_less_frequency)
            data["IP"].append(ip_less_frequency)

    return pd.DataFrame(data)

# Generate the data
n = 100
num_cards = 10
df = generate_data(n, num_cards)
df.to_csv('IrrelevantSource_DataSet.csv', index=False)

# Read the CSV file
df = pd.read_csv('IrrelevantSource_DataSet.csv', dtype=str)

# Identify records to mark as NaN
# Create a DataFrame for records with same card_number, Phone 1, Phone 2 but different email and IP
grouped = df.groupby(['card_number', 'Phone 1', 'Phone 2'])

def mark_less_frequent_emails(group):
    # Find the most frequent email within the group
    email_counts = group['Email'].value_counts()
    if email_counts.empty:
        return group
    most_frequent_email = email_counts.idxmax()

    # Mark emails as NaN where they are less frequent compared to the most frequent one
    group.loc[(group['Email'] != most_frequent_email), 'Email'] = np.nan
    return group

df_with_nan_emails = grouped.apply(mark_less_frequent_emails).reset_index(drop=True)

# Save the modified DataFrame to a new CSV file
df_with_nan_emails.to_csv('Relevant_DataSet.csv', index=False)

print("Data generation complete. Saved to 'IrrelevantSource_DataSet.csv' and 'Relevant_DataSet.csv'.")
