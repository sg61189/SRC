import os
import json
import argparse

def restoreBackup(random_name):
    json_path = os.path.join("../BACKUP", "backup_tracker.json")

    if not os.path.exists(json_path):
        print("Error: Backup tracker file not found!")
        return

    with open(json_path, "r") as json_file:
        tracker_data = json.load(json_file)

    # Check if the random_name exists in the tracker
    if random_name not in tracker_data:
        print(f"Error: No backup found for the file name: {random_name}")
        return

    # Get the backup details
    backup_info = tracker_data[random_name]
    backup_path = backup_info["backup_path"]
    original_path = backup_info["original_path"]

    # Ensure the backup file exists
    if not os.path.exists(backup_path):
        print(f"Error: Backup file not found at: {backup_path}")
        return

    os.makedirs(os.path.dirname(original_path), exist_ok=True)
    with open(backup_path, "r") as backup_file:
        backup_content = backup_file.read()
    with open(original_path, "w") as original_file:
        original_file.write(backup_content)

    print(f"Backup restored successfully to: {original_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Restore a backup file to its original location."
    )
    parser.add_argument(
        "random_name",
        type=str,
        help="The randomized name of the backup file to restore."
    )

    args = parser.parse_args()
    restoreBackup(args.random_name)

if __name__ == "__main__":
    main()
