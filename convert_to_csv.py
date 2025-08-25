import csv

def convert_census_to_csv():
    columns = []
    with open('census-bureau.columns', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                columns.append(line)
    
    with open('census-bureau.data', 'r') as infile, open('census_data.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(columns)
        
        for line in infile:
            line = line.strip()
            if line:
                row = line.split(',')
                writer.writerow(row)

if __name__ == "__main__":
    convert_census_to_csv()