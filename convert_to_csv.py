import csv
import os

def convert_census_to_csv():
    columns_file = './source_data/census-bureau.columns'
    data_file = './source_data/census-bureau.data'
    output_file = 'census_data.csv'


    if not os.path.exists(columns_file):
        raise FileNotFoundError(f"Column definitions file not found: {columns_file}")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    columns = []
    with open(columns_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                columns.append(line)
    
    row_count = 0
    with open(data_file, 'r') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(columns)
        for line in infile:
            line = line.strip()
            if line:
                row = line.split(',')
                writer.writerow(row)
                row_count += 1
    
    return output_file, row_count, len(columns)

if __name__ == "__main__":
    try:
        convert_census_to_csv()
    except Exception as e:
        print(f"Error during conversion: {e}")