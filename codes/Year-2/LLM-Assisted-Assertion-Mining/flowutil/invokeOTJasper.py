import subprocess
import re
import csv
import argparse

# Patterns to extract tables
fpv_pattern = r"\\|\\s+name\\s+\\|.*?\\n(\\|.*?\\|(?:\\n|$)+)"
coverage_pattern = r"\\|\\s+formal\\s+\\|.*?\\n(\\|.*?\\|(?:\\n|$)+)"

# Helper function to parse table text to list
def parse_table(table_text):
    lines = table_text.strip().split("\\n")
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    rows = [
        [cell.strip() for cell in row.split("|") if cell.strip()]
        for row in lines[1:]
    ]
    return headers, rows

# Extract tables and write to CSV
def extract_and_write(pattern, log_content, output_csv):
    match = re.search(pattern, log_content, re.DOTALL)
    if match:
        table_text = match.group(1)
        headers, rows = parse_table(table_text)

        # Write to CSV
        with open(output_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Extracted table written to {output_csv}.")
    else:
        print(f"No table found for pattern in {output_csv}.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run dvsim and extract FPV and coverage metrics."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="The configuration name to pass to dvsim (e.g., prim_packer_fpv)."
    )
    parser.add_argument(
        "--repo_top",
        required=True,
        help="Path to the repository top-level directory ($REPO_TOP)."
    )

    args = parser.parse_args()

    # Build the command
    command = [
        f"{args.repo_top}/util/dvsim/dvsim.py",
        f"{args.repo_top}/hw/top_earlgrey/formal/top_earlgrey_fpv_prim_cfgs.hjson",
        "--select-cfgs",
        args.config,
    ]

    # Run the command and capture its output
    try:
        print("Running command to capture log...")
        process = subprocess.run(
            " ".join(command),
            capture_output=True,
            text=True,
            shell=True  # Use shell=True for $REPO_TOP expansion
        )
        log_content = process.stdout

        if process.returncode != 0:
            print("Error while running the command:")
            print(process.stderr)
            return

        print("Command executed successfully. Processing logs...")

        # Extract FPV Results table
        extract_and_write(fpv_pattern, log_content, "metrics_FPV.csv")

        # Extract Coverage Results table
        extract_and_write(coverage_pattern, log_content, "metrics_Coverage.csv")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

