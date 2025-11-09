# data_preprocessing.py
import pandas as pd
import ast
from logger_utils import setup_logger, update_process_id, logging

config, pid = setup_logger()

logging.info("Data Preprocessing Process Started")

def label_from_rating(r):
    r = float(r)
    if r >= 4: return "positive "
    if r <= 2: return "negative "
    return "neutral "

def parse_helpful(x):
    try:
        if isinstance(x, str):
            x = ast.literal_eval(x)
        up, total = x
        return 0.0 if total == 0 else up / total
    except Exception:
        return 0.0

def preprocess(input_file, output_file):
    try:
        logging.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        logging.info("Generating sentiment labels")
        df["label"] = df["overall"].apply(label_from_rating)

        logging.info("Combining text fields")
        df["text"] = (
            df["summary"].fillna("").astype(str) + " " +
            df["reviewText"].fillna("").astype(str)
        ).str.lower()

        logging.info("Parsing helpfulness ratio and review length")
        df["helpful_ratio"] = df["helpful"].apply(parse_helpful)
        df["review_len"] = df["reviewText"].fillna("").astype(str).str.len()

        df.to_csv(output_file, index=False)
        logging.info(f"Saved proceed data to {output_file}")
        logging.info("Data Preprocessing Process Ended")
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
    finally:
        update_process_id(config, current_pid=pid)

if __name__ == "__main__":
    preprocess("data/row/Instruments_Reviews.csv", "data/processed_data/cleaned_reviews.csv")
