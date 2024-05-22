from pdf2docx import Converter

INPUT_FILE = 'Albright, Charles.pdf'
OUTPUT_DOCX_FILE = 'data/Albright_Charles.docx'

def convert_pdf_to_docx(input_file, output_file):
    cv = Converter(input_file)
    cv.convert(output_file, start=0, end=None)
    cv.close()

convert_pdf_to_docx(INPUT_FILE, OUTPUT_DOCX_FILE)
print(f'PDF converted to DOCX and saved to {OUTPUT_DOCX_FILE}')

import docx
import pandas as pd

OUTPUT_CSV_FILE = 'data/Albright_Charles_table.csv'
INPUT_DOCX_FILE = 'data/Albright_Charles.docx'


def extract_tables_from_docx(docx_file, output_csv_file):
    doc = docx.Document(docx_file)
    all_tables = []

    for table in doc.tables:
        data = []
        keys = None
        for i, row in enumerate(table.rows):
            text = [cell.text.strip() for cell in row.cells]
            if i == 0:
                keys = text
                continue
            row_data = dict(zip(keys, text))
            data.append(row_data)
        if data:
            all_tables.append(pd.DataFrame(data))

    # Concatenate all tables into a single DataFrame
    if all_tables:
        combined_df = pd.concat(all_tables, ignore_index=True)
        combined_df.to_csv(output_csv_file, index=False)
        print(f'Tables extracted and saved to {output_csv_file}')
    else:
        print('No tables found in the DOCX.')

extract_tables_from_docx(INPUT_DOCX_FILE, OUTPUT_CSV_FILE)


# Path to the cleaned CSV file
cleaned_csv_file_path = 'data/Albright_Charles_table.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(cleaned_csv_file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Optionally rename columns if necessary
# df.columns = ['Date', 'Age', 'Life Event']

# Display the columns
print(df.columns)

# Display the first few rows of the cleaned DataFrame
print(df.head())

# Save the cleaned DataFrame to a new CSV file if needed
output_cleaned_csv_path = 'data/Cleaned_Albright_Charles_table.csv'
df.to_csv(output_cleaned_csv_path, index=False)

print(f'Cleaned data saved to {output_cleaned_csv_path}')