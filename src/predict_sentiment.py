# predict.py
import joblib
import pandas as pd
from scipy.sparse import hstack
from logger_utils import setup_logger, update_process_id, logging

config, pid = setup_logger()
logging.info("Prediction Process Started")

def predict_sentiment(text):
    try:
        logging.info("Loading model and vectorizer")
        model = joblib.load("artifacts/sentiment_model.joblib")
        tfidf = joblib.load("artifacts/vectorizer.joblib")

        X_text = tfidf.transform([text.lower()])
        num = pd.DataFrame({
            "helpful_ratio": [0],
            "review_len": [len(text)]
        }).values

        X = hstack([X_text, num])
        pred = model.predict(X)
        logging.info(f"Prediction complete: {pred[0]}")
        logging.info("Prediction Process Ended")
        return pred[0]
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None
    finally:
        update_process_id(config, current_pid=pid)

if __name__ == "__main__":
    print(predict_sentiment("Amazing product, works great!"))
