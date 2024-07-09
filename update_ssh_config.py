"""
This script takes in a single argument: a 3-digit code like "003", "089" or "113" and edits the ~/.ssh/config file
and edits this section of the file:

Host omnode
  HostName nodeXXX
  ProxyJump om
  User yourusername

where XXX is some existing 3-digit code. We want to replace the three characters in the position XXX with the new 3-digit code
taken in as an argument to this script.


Usage: python update_ssh_config.py 107
107 is an example of omnode digits
"""
#!/usr/bin/env python3

import sys
import os
import re

def backup_config_file(config_path):
    """Create a backup of the config file."""
    backup_path = config_path + ".backup"
    with open(config_path, "r") as src, open(backup_path, "w") as dst:
        dst.writelines(src.readlines())
    print(f"Backup created at: {backup_path}")

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: omnode <3-digit-code>")
        sys.exit(1)

    new_code = sys.argv[1]
    if len(new_code) != 3 or not new_code.isdigit():
        print("Usage: omnode <3-digit-code>")
        sys.exit(1)

    config_path = os.path.join(os.path.expanduser('~'), ".ssh", "config")
    try:
        # Create a backup before making changes
        backup_config_file(config_path)

        # Read in the file
        with open(config_path, "r") as f:
            lines = f.readlines()

        # Update the relevant line
        updated = False
        for i, line in enumerate(lines):
            if "Host omnode" in line:
                for j in range(i+1, len(lines)):
                    if re.search(r"HostName node\d{3}", lines[j]):
                        lines[j] = re.sub(r"node\d{3}", "node" + new_code, lines[j])
                        updated = True
                        break
            if updated:
                break

        # Write the file out again
        with open(config_path, "w") as f:
            f.writelines(lines)

    except FileNotFoundError:
        print(f"Error: File {config_path} not found.")
    except PermissionError:
        print(f"Error: No permission to read or write to {config_path}.")
    except Exception as e:
        print(f"Unexpected error: {e}")

