# logger_utils.py
import logging
import json
import os

def setup_logger(config_file="config.json", log_file="Log/logfile.txt"):
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Load config
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        config = {"CURRENT_PROCESS_ID": 1}

    current_pid = config["CURRENT_PROCESS_ID"]

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format=f"PID:{current_pid} - %(asctime)s - %(levelname)s - %(message)s"
    )

    # Return config and PID for caller
    return config, current_pid


def update_process_id(config, config_file="config.json", current_pid=0):
    config["CURRENT_PROCESS_ID"] = current_pid + 1
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
