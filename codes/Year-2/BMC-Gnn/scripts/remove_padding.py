import os
import re

base_directory = '/home/src2024/bmc-gnn2/data/chosen_circuits_non-inductive'

def remove_padded_zeros(filename):
    match = re.match(r'(.*)_(0*)(\d+)\.pkl', filename)
    if match:
        base = match.group(1)
        number = match.group(3)
        new_name = f'{base}_{number}.pkl'
        return new_name
    return filename

for subdir in os.listdir(base_directory):
    subdirectory_path = os.path.join(base_directory, subdir)
    
    if os.path.isdir(subdirectory_path):
        print(f"Processing files in {subdirectory_path}...")
        
        for filename in os.listdir(subdirectory_path):
            if filename.endswith('.pkl'):
                old_path = os.path.join(subdirectory_path, filename)
                new_name = remove_padded_zeros(filename)
                if new_name != filename:
                    new_path = os.path.join(subdirectory_path, new_name)
                    os.rename(old_path, new_path)
                    print(f'Renamed: {old_path} to {new_path}')

print("All files processed.")
