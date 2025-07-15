from sklearn.metrics import classification_report
import logging
import json

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """Generate evaluation metrics"""
    try:
        # Generate predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = {
            "accuracy": report['accuracy'],
            "precision": report['1']['precision'],
            "recall": report['1']['recall'],
            "f1": report['1']['f1-score'],
            "roc_auc": None  # Will calculate if needed
        }
        
        logger.info("Model evaluation completed")
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
