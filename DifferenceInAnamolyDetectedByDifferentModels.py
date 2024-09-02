import pandas as pd


def load_csv(file):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file)


def find_differences(df1, df2, df3):
    """Find records that are not the same across three DataFrames."""
    # Combine all records into a single DataFrame with an indicator of source
    combined_df = pd.concat([df1, df2, df3], keys=['File1', 'File2', 'File3']).reset_index(level=0)

    # Group by all columns and filter out groups with identical records across all sources
    difference_df = combined_df.groupby(list(combined_df.columns[1:])).filter(lambda x: len(x['level_0'].unique()) > 1)

    return difference_df


def save_differences(difference_df, output_file='difference.csv'):
    """Save the differences DataFrame to a CSV file."""
    difference_df.to_csv(output_file, index=False)


def main(file1, file2, file3):
    # Load the CSV files
    df1 = load_csv(file1)
    df2 = load_csv(file2)
    df3 = load_csv(file3)

    # Find the differences
    differences = find_differences(df1, df2, df3)

    # Save the differences to a file
    if not differences.empty:
        save_differences(differences)
        print(f"Differences found and saved to 'difference.csv'.")
    else:
        print("No differences found across the files.")


if __name__ == "__main__":
    # Example usage
    file1 = 'denoising_anomalies.csv'
    file2 = 'anomalies.csv'
    file3 = 'anomalies_vae.csv'
    main(file1, file2, file3)
