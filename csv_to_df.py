import pandas as pd

# Path to the CSV file
csv_file_path = 'data/Albright_Charles_table.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame to inspect the structure
print(df.head())

# Example of inspecting and cleaning up the DataFrame
# Display the columns
print(df.columns)

# Rename columns if needed
# df.columns = ['Date', 'Age', 'Life Event', 'Other Columns']

# Display the first few rows of the cleaned DataFrame
print(df.head())

# Path to save the cleaned CSV file
cleaned_csv_file_path = 'data/Cleaned_Albright_Charles_table.csv'

# Save the cleaned DataFrame to a CSV file
df.to_csv(cleaned_csv_file_path, index=False)

print(f'Cleaned data saved to {cleaned_csv_file_path}')