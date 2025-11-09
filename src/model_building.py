# model_building.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from logger_utils import setup_logger, update_process_id, logging
import yaml

config, pid = setup_logger()

logging.info("Model Building Process Started")

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

max_iter = params["model_building"]["model_params"]["max_iter"]
C = params["model_building"]["model_params"]["C"]
class_weight = params["model_building"]["model_params"]["class_weight"]


def train():
    try:
        logging.info("Loading feature data")
        X_train = joblib.load("artifacts/X_train.joblib")
        X_test  = joblib.load("artifacts/X_test.joblib")
        y_train = joblib.load("artifacts/y_train.joblib")
        y_test  = joblib.load("artifacts/y_test.joblib")

        logging.info("Training Logistic Regression model")
        model = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        logging.info(f"Model evaluation: Accuracy={acc:.4f}, F1-macro={f1:.4f}")
        logging.info(f"\n{classification_report(y_test, preds)}")

        joblib.dump(model, "artifacts/sentiment_model.joblib")
        logging.info("Model saved successfully")
        logging.info("Model Building Process Ended")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
    finally:
        update_process_id(config, current_pid=pid)

if __name__ == "__main__":
    train()
