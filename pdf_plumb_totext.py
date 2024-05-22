import pdfplumber
import pandas as pd

INPUT_FILE = 'Albright, Charles.pdf'
OUTPUT_FILE = 'data/2_Albright_Charles_table.csv'

# Open the PDF file
with pdfplumber.open(INPUT_FILE) as pdf:
    all_tables = []
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            all_tables.append(table)

    # Assuming the table you want is the first one
    if all_tables:
        df = pd.DataFrame(all_tables[0][1:], columns=all_tables[0][0])
        df.to_csv(OUTPUT_FILE, index=False)
        print(f'Table extracted and saved to {OUTPUT_FILE}')
    else:
        print('No tables found in the PDF.')