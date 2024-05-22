# convert pdfs to data frames

import tabula
import pandas as pd

INPUT_FILE = 'Albright, Charles.pdf'
OUTPUT_FILE = 'data/Albright_Charles_table.csv'

# Read the table from the PDF
tables = tabula.read_pdf(INPUT_FILE, pages='all', multiple_tables=True)

# Assuming the table you want is the first one
if tables:
    df = tables[0]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f'Table extracted and saved to {OUTPUT_FILE}')
else:
    print('No tables found in the PDF.')




