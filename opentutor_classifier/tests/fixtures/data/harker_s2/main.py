#Just a script to open up each of the csvs and change TRUE FALSE 
#Got this script from chatgpt o_o!

import os
import csv
import re

def process_csv_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                
                # Read the CSV file
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
                
                # Modify the last column
                for i, row in enumerate(rows):
                    if i == 0: continue
                    row[0] = int(row[0]) - 1
                    #row[1] = "'" + row[1] + "'"
                    #row[1] = re.sub("\'", "", row[1])
                    if row[-1] == 'true':
                        row[-1] = 'good'
                    elif row[-1] == 'false':
                        row[-1] = 'bad'
                
                # Overwrite the original file with the modified data
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)
                
                print(f"{file_path} processed successfully.")

    print("All CSV files processed.")


if __name__ == "__main__":
    directory = os.getcwd()  # Get the current directory
    process_csv_files(directory)
