import requests
from logger_utils import setup_logger, update_process_id, logging
import yaml

config, pid = setup_logger()
logging.info("Data ingestion Process Stated")

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

url = params["data_ingestion"]["url"]

try:
    logging.info("Starting download")
    response = requests.get(url)

    if response.status_code == 200:
        with open("data/row/Instruments_Reviews.csv", "wb") as f:
            f.write(response.content)
        logging.info("Downloaded successfully")
        logging.info("Data ingestion Process Ended")
    else:
        logging.error(f"Failed {response.status_code}")
except Exception as e:
    logging.error(f"Error: {e}")
finally:
    update_process_id(config, current_pid=pid)
