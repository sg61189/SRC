import json
import os
import pathlib

def find_file(filename, search_path):
    """
    Recursively search for a file in the given path and its subfolders
    Returns the found path or None if not found
    """
    for root, _, files in os.walk(search_path):
        if filename in files:
            return os.path.relpath(os.path.join(root, filename))
    return None

# Read the JSON file
with open("IPconfig.json", "r") as f:
    data = json.load(f)

# Base search directory
base_dir = "../../hw/"

# Process each config entry
processed_data = []
for i in data['use_cfgs']:
    if i['cov']:
        # Extract the base filenames
        rtl_filename = f"{i['name']}.sv"
        spec_filename = f"{i['name']}.md"
        
        # Search for the actual files
        rtl_path = find_file(rtl_filename, base_dir)
        spec_path = find_file(spec_filename, base_dir)
        
        # Use found paths if they exist, otherwise use default paths
        rtl_final_path = rtl_path if rtl_path else i['rel_path'].replace("{sub_flow}/{tool}", f"rtl/{i['name']}.sv")
        spec_final_path = spec_path if spec_path else i['rel_path'].replace("{sub_flow}/{tool}", f"doc/{i['name']}.md")
        
        processed_data.append({
            "name": i['name'],
            "rtl": rtl_final_path,
            "spec": spec_final_path
        })

# Write the processed data back to the file
with open("IPconfig2.json", "w") as f:
    f.write(json.dumps({"config": processed_data}, indent=4))