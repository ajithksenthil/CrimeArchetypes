import csv
import datetime

def read_csv_life_events(file_path):
    events = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row.get('Date', '')
            age = row.get('Age', '')
            event = row.get('Life Event', '')
            if event:  # Only add non-empty events
                events.append({
                    'date': date,
                    'age': age,
                    'event': event
                })
    return events
