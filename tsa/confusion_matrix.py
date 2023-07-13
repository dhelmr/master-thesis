from dataclasses import dataclass
from sklearn.metrics import confusion_matrix


@dataclass
class ConfusionMatrix:
    """
    Data Class for the values of a confusion matrix
    """
    tp: int
    tn: int
    fp: int
    fn: int

    @staticmethod
    def from_predictions(pred_y, ground_truth_y, labels=None):
        tn, fp, fn, tp = confusion_matrix(ground_truth_y, pred_y, labels=labels).ravel()
        return ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp)

    def calc_unweighted_measurements(cm) -> dict:
        """
        Calculates different metrics for the values of a confusion matrix.
        For terminology see https://en.wikipedia.org/wiki/Precision_and_recall
        """
        tp, fp, fn, tn = cm.tp, cm.fp, cm.fn, cm.tn
        sd = safe_divide
        metrics = dict()
        p = tp + fn
        n = tn + fp
        metrics["precision"] = sd(tp, (tp + fp))
        metrics["recall"] = sd(tp, p)
        metrics["f1_score"] = sd(2 * tp, (2 * tp + fp + fn))
        metrics["accuracy"] = sd((tp + tn), (p + n))
        metrics["positives"] = p
        metrics["negatives"] = n
        metrics["tnr"] = sd(tn, n)
        metrics["npv"] = sd(tn, (tn + fn))
        metrics["fpr"] = sd(fp, n)
        metrics["fdr"] = sd(fp, (fp + tp))
        metrics["for"] = sd(fn, (fn + tn))
        metrics["fnr"] = sd(fn, (fn + tp))
        metrics["balanced_accuracy"] = (metrics["recall"] + metrics["tnr"]) / 2
        metrics["true_negatives"] = tn
        metrics["true_positives"] = tp
        metrics["false_negatives"] = fn
        metrics["false_positives"] = fp
        metrics["kappa"] = 1 - sd(
            1 - metrics["accuracy"],
            1 - sd((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), pow(p + n, 2)),
        )
        metrics["mcc"] = sd(
            tp * tn - fp * fn, pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5)
        )
        metrics["support"] = n + p
        return metrics


def safe_divide(q1, q2) -> float:
    try:
        value = q1 / q2
    except ZeroDivisionError:
        value = float("NaN")
    except RuntimeWarning:
        value = float("NaN")
    return value