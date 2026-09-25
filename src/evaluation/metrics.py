from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    brier_score_loss
)


def compute_metrics(y_true, y_prob):
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    return {
        "pr_auc": pr_auc,
        "brier_score": brier,
        "precision_curve": precision,
        "recall_curve": recall,
        "thresholds": thresholds
    }
