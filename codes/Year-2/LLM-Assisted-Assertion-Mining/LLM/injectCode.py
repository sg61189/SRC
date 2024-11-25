import argparse
import datetime
import re
import os
import json
import random
import string

def createBackUP(RTL):
    rtl_path = os.path.abspath(RTL)
    rtl_name = os.path.basename(rtl_path)

    with open(rtl_path, "r") as f:
        rtl = f.read()

    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=15)) + ".sv"
    backup_dir = os.path.join("../BACKUP")
    code_dir = os.path.join(backup_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    backup_path = os.path.join(code_dir, random_name)

    with open(backup_path, "w") as f:
        f.write(rtl)

    # Path for the JSON tracking file
    json_path = os.path.join(backup_dir, "backup_tracker.json")

    # Load the existing JSON data if the file exists
    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            tracker_data = json.load(json_file)
    else:
        tracker_data = {}

    # Add the new backup entry to the tracker
    tracker_data[random_name] = {
        "original_file": rtl_name,
        "original_path": rtl_path,
        "backup_path": os.path.abspath(backup_path)
    }

    with open(json_path, "w") as json_file:
        json.dump(tracker_data, json_file, indent=4)

    print(f"Backup created: {backup_path}")
    print(f"Tracker updated: {json_path}")

def replaceAndInject(RTL, assertions):
    with open(RTL, "r") as f:
        RTL = f.read()
    with open(assertions, "r") as f:
        assertions = json.load(f)
    
    # The pattern
    pattern = (
        r'^\s*\/\/+\n'
        r'\s*\/\/\s*Assertions,\s*Assumptions,\s*and\s*Coverpoints\s*\/\/\n'
        r'\s*\/\/+\n'
        r'\s*\/\/\s*Assumption:\s*mask_i\s*should\s*be\s*contiguous\s*ones\s*\n'
    )
    
    # Create the full pattern - using raw string for the entire pattern
    full_pattern = r'(' + pattern + r')(.*?)(\n\s*endmodule)'
    
    # Replace the matched content with just endmodule
    result = re.sub(
        full_pattern,
        r'\3',  # Keep only the endmodule part
        RTL,
        flags=re.MULTILINE | re.DOTALL
    )
    
    result = result.replace("endmodule", assertions['code'] + "endmodule")
    return result

def saveCode(original, new):
    with open(original, "w") as f:
        f.write(new)
    print(f"File Updated to {original}")

def main():
    parser = argparse.ArgumentParser(description="Reads assertions from a JSON file and Injects them to a RTL. Also, creates one backup of the original RTL with a specific hash to be found at project root directory BACKUP.")
    parser.add_argument("-r", "--rtl", dest="rtl", required=True, help="RTL File")
    parser.add_argument("-ast", "--assertions", dest="asts", required=True, help="JSON file path for generated assertions")
    
    args = parser.parse_args()
    createBackUP(args.rtl)
    code = replaceAndInject(args.rtl, args.asts)
    saveCode(args.rtl, code)
if __name__ == "__main__":
    main()