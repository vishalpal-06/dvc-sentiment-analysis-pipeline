# feature_engineering.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from logger_utils import setup_logger, update_process_id, logging
import yaml

config, pid = setup_logger()

with open("params.yaml","r") as f:
    params = yaml.safe_load(f)


test_size = params["feature_engineering"]["vectorizer_params"]["test_size"]
random_state = params["feature_engineering"]["vectorizer_params"]["random_state"]
ngram_range = params["feature_engineering"]["vectorizer_params"]["ngram_range"]
min_df = params["feature_engineering"]["vectorizer_params"]["min_df"]
max_feature = params["feature_engineering"]["vectorizer_params"]["max_features"]



logging.info("Feature Engineering Process Started")

def build_features(input_file, out_X_train, out_X_test, out_y_train, out_y_test):
    try:
        logging.info(f"Reading cleaned data from {input_file}")
        df = pd.read_csv(input_file)

        logging.info("Splitting data into train/test")
        X_train, X_test, y_train, y_test = train_test_split(
            df, df["label"], test_size=test_size, random_state=random_state, stratify=df["label"]
        )

        logging.info("Building TF-IDF features")
        tfidf = TfidfVectorizer(ngram_range=tuple(ngram_range), min_df=min_df, max_features=max_feature)
        X_train_text = tfidf.fit_transform(X_train["text"])
        X_test_text = tfidf.transform(X_test["text"])

        logging.info("Adding numeric features")
        X_train_num = X_train[["helpful_ratio", "review_len"]].values
        X_test_num = X_test[["helpful_ratio", "review_len"]].values

        X_train_all = hstack([X_train_text, X_train_num])
        X_test_all = hstack([X_test_text, X_test_num])

        logging.info("Saving feature matrices and vectorizer")
        joblib.dump(tfidf, "artifacts/vectorizer.joblib")
        joblib.dump(X_train_all, out_X_train)
        joblib.dump(X_test_all, out_X_test)
        joblib.dump(y_train, out_y_train)
        joblib.dump(y_test, out_y_test)

        logging.info("Feature Engineering Process Ended")
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
    finally:
        update_process_id(config, current_pid=pid)

if __name__ == "__main__":
    build_features(
        "data/Processed_data/cleaned_reviews.csv",
        "artifacts/X_train.joblib",
        "artifacts/X_test.joblib",
        "artifacts/y_train.joblib",
        "artifacts/y_test.joblib",
    )
