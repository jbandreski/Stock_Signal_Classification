import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)

LABEL_NAMES = ["Hold", "Buy", "Sell"]

def classification_report(model, X_test: np.ndarray, y_true: np.ndarray):
    """Print accuracy, precision, recall, F1, and plot confusion matrix."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        y_pred = logits.argmax(dim=1).numpy()

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()

    return y_pred
