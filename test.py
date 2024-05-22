import pandas as pd

# Path to the Excel file
excel_file_path = 'mnt/data/Albright_Charles.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path, sheet_name='Combined Data', header=None)

# Display the first few rows of the DataFrame to inspect its structure
print("Original DataFrame:")
print(df.head(10))

# Initialize lists to hold the rows for each table
type1_rows = []
type2_rows = []
current_section = None

# Function to identify if a row is the heading for "Type 1" data
def is_type1_heading(row):
    return row.str.contains('Date').any() and row.str.contains('Age').any() and row.str.contains('Life Event').any()

# Function to identify if a row is the heading for "Type 2" data
def is_type2_heading(row):
    return row.str.contains('General Information').any()

# Iterate over the rows to separate data into different sections
for index, row in df.iterrows():
    if is_type1_heading(row):
        current_section = 'type1'
        type1_rows.append(row.values.tolist())
    elif is_type2_heading(row):
        current_section = 'type2'
        type2_rows.append(['Heading', 'Value'])  # Add the headers for type2
    elif current_section == 'type1':
        type1_rows.append(row.values.tolist())
    elif current_section == 'type2':
        # Ensure we only have two columns per row
        row_values = [v for v in row.values if pd.notna(v)]
        if len(row_values) == 2:
            type2_rows.append(row_values)
        elif len(row_values) > 2:
            # Join the extra columns into one, but ensure no duplication
            type2_rows.append([row_values[0], ' '.join(map(str, row_values[1:]))])

# Convert the lists to DataFrames
if type1_rows:
    type1_df = pd.DataFrame(type1_rows[1:], columns=type1_rows[0])
else:
    type1_df = pd.DataFrame()

if type2_rows:
    type2_df = pd.DataFrame(type2_rows[1:], columns=type2_rows[0])
else:
    type2_df = pd.DataFrame()

# Clean the Type 2 DataFrame to remove duplicate headings in the value column
def clean_value(row):
    heading = row['Heading']
    value = row['Value']
    if heading in value:
        value = value.replace(heading, '').strip()
    return value

if not type2_df.empty:
    type2_df['Value'] = type2_df.apply(clean_value, axis=1)

# Display the cleaned DataFrames
print("Type 1 DataFrame (Date, Age, Life Event):")
print(type1_df.head())

print("Type 2 DataFrame (General Information, etc.):")
print(type2_df.head())

# Save the cleaned DataFrames to CSV
type1_df.to_csv('mnt/data/Type1_Cleaned_Albright_Charles_table.csv', index=False)
type2_df.to_csv('mnt/data/Type2_Cleaned_Albright_Charles_table.csv', index=False)

print(f'Type 1 DataFrame saved to mnt/data/Type1_Cleaned_Albright_Charles_table.csv')
print(f'Type 2 DataFrame saved to mnt/data/Type2_Cleaned_Albright_Charles_table.csv')