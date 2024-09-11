
import numpy as np
import torch

def compute_score(confusion_matrix):
    # Assuming confusion_matrix is a 2D tensor
    TP = torch.diag(confusion_matrix)
    FP = confusion_matrix.sum(dim=0) - TP
    FN = confusion_matrix.sum(dim=1) - TP
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Compute TPR, SPC, PPV, NPV, FPR, FDR, FNR, DE
    TPR = torch.nan_to_num(TP / (TP + FN))
    SPC = torch.nan_to_num(TN / (FP + TN))
    PPV = torch.nan_to_num(TP / (TP + FP))
    NPV = torch.nan_to_num(TN / (TN + FN))
    FPR = torch.nan_to_num(FP / (FP + TN))
    FDR = torch.nan_to_num(FP / (FP + TP))
    FNR = torch.nan_to_num(FN / (FN + TP))
    DF = torch.nan_to_num(TPR * SPC)

    # Compute precision, recall, and F1 score (micro)
    precision_micro = TP.sum() / (TP.sum() + FP.sum())
    recall_micro = TP.sum() / (TP.sum() + FN.sum())
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

    results = {
        "TPR": TPR.mean(),  # mean for multi-class
        "SPC": SPC.mean(),
        "PPV": PPV.mean(),
        "NPV": NPV.mean(),
        "FPR": FPR.mean(),
        "FDR": FDR.mean(),
        "FNR": FNR.mean(),
        "DE": DF.mean(),  # Diagnostic Efficiency
        "F1_micro": f1_micro  # F1 score (micro)
    }
    
    return results

def calculate_metrics_from_confusion_matrix(confusion_matrix):
    tp = confusion_matrix.diagonal()
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp

    precision = tp / (tp + fp + 1e-6)  # Adding a small value to avoid division by zero
    recall = tp / (tp + fn + 1e-6)     # Adding a small value to avoid division by zero
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)  # Adding a small value to avoid division by zero

    return precision, recall, f1_score

def compute_macro_weighted_averages(precision, recall, f1_score, support):
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()

    weighted_precision = (precision * support).sum() / support.sum()
    weighted_recall = (recall * support).sum() / support.sum()
    weighted_f1 = (f1_score * support).sum() / support.sum()

    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def classification_report_from_cm(confusion_matrix, class_names, digits=2):
    precision, recall, f1_score = calculate_metrics_from_confusion_matrix(confusion_matrix)

    accuracy = confusion_matrix.diagonal().sum() / confusion_matrix.sum()

    support = confusion_matrix.sum(axis=1)

    averages = compute_macro_weighted_averages(precision, recall, f1_score, support)

    report_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': confusion_matrix,
        'macro_precision': averages['macro_precision'],
        'macro_recall': averages['macro_recall'],
        'macro_f1': averages['macro_f1'],
        'weighted_precision': averages['weighted_precision'],
        'weighted_recall': averages['weighted_recall'],
        'weighted_f1': averages['weighted_f1'],
        'support': support
    }

    headers = ["precision", "recall", "f1-score", "support"]
    longest_last_line_heading = "weighted avg"
    name_width = max(len(cn) for cn in class_names)
    width = max(name_width, len(longest_last_line_heading), digits)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for i, class_name in enumerate(class_names):
        report += row_fmt.format(class_name, precision[i], recall[i], f1_score[i], int(support[i]), width=width, digits=digits)
    report += "\n"
    
    report += "{:>{width}s} {:>9} {:>9} {:>9.{digits}f} {:>9}\n".format(
        "accuracy", "", "", accuracy, int(support.sum()), width=width, digits=digits
    )
    report += "{:>{width}s} {:>9.{digits}f} {:>9.{digits}f} {:>9.{digits}f} {:>9}\n".format(
        "macro avg", averages['macro_precision'], averages['macro_recall'], averages['macro_f1'], int(support.sum()), width=width, digits=digits
    )
    report += "{:>{width}s} {:>9.{digits}f} {:>9.{digits}f} {:>9.{digits}f} {:>9}\n".format(
        "weighted avg", averages['weighted_precision'], averages['weighted_recall'], averages['weighted_f1'], int(support.sum()), width=width, digits=digits
    )
    
    return report
