import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar



# Clopper-Pearson method for ACC, SEN, SPE CIs
def exact_clopper_pearson(n_successes, n_total, alpha=0.05):
    '''
    Compute the exact Clopper-Pearson confidence interval for a binomial proportion.
    
    Args:
        n_successes (int): Number of successes (e.g., TP or TN).
        n_total (int): Total number of trials (e.g., TP + FN).
        alpha (float): Significance level (default = 0.05 for 95% CI).
    
    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval.
    '''
    lower, upper = proportion_confint(n_successes, n_total, alpha=alpha, method='beta')
    return lower, upper


# Delong method for AUC CIs
def delong_auc_ci(y_true, y_pred, alpha=0.05):
    '''
    Approximate the AUC confidence interval using a standard error-based DeLong method.
    
    Args:
        y_true (array-like, shape: n_samples, n_classes): Binary ground truth labels.
        y_pred (array-like, shape: n_samples, n_classes): Continuous prediction scores.
        alpha (float): Significance level for CI (default = 0.05).
    
    Returns:
        tuple: (lower_bound, upper_bound) of the AUC confidence interval.
    '''
    auc = roc_auc_score(y_true, y_pred)
    n = len(y_true)
    se_auc = np.sqrt((auc * (1 - auc)) / n)
    z = st.norm.ppf(1 - alpha / 2)
    lower = auc - z * se_auc
    upper = auc + z * se_auc
    return lower, upper


# Bootstrap method for CIs (generic method)
def bootstrap_ci(metric_fn, y_true, y_pred, num_bootstraps, alpha=0.05, threshold=0.5):
    '''
    Compute confidence intervals for a binary metric using bootstrapping.
    
    Args:
        metric_fn (function): Metric function that returns a scalar.
        y_true (array-like, shape: n_samples, n_classes): Ground truth binary labels.
        y_pred (array-like, shape: n_samples, n_classes): Continuous predictions.
        num_bootstraps (int): Number of bootstrap samples.
        alpha (float): Significance level for CI.
        threshold (float): Threshold to convert scores to binary predictions.
    
    Returns:
        tuple: (lower_bound, upper_bound) of the bootstrap confidence interval.
    '''

    stats = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(num_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        # Threshold predictions for binary metrics
        stat = metric_fn(y_true[indices], (y_pred[indices] >= threshold).astype(int))
        stats.append(stat)
    lower = np.percentile(stats, (alpha / 2) * 100)
    upper = np.percentile(stats, (1 - alpha / 2) * 100)
    return lower, upper


# Calculate and report metrics
def calculate_metrics(
    y_true, y_pred, metrics_to_report,
    label_names=None, threshold=0.5,
    round_decimals=2, display_as_table=True,
    num_bootstraps=10, alpha=0.05):
    '''
    Compute classification metrics (per class and aggregated) with confidence intervals.
    
    Args:
        y_true (np.ndarray): Binary ground truth matrix (n_samples, n_classes).
        y_pred (np.ndarray): Continuous prediction matrix (n_samples, n_classes).
        metrics_to_report (list): Metrics to compute. Choose from: ["accuracy", "precision", "recall", "specificity", "f1", "auc", "ppv", "npv"].
        label_names (list, optional): Class label names.
        threshold (float): Threshold to binarize predictions.
        round_decimals (int): Decimal precision for reported values.
        display_as_table (bool): Whether to print a formatted summary table.
        num_bootstraps (int): Bootstrap iterations for CIs.
        alpha (float): Significance level for CIs.
    
    Returns:
        dict: Dictionary of metrics and confidence intervals for each label, including macro- and micro-averages.
    '''

    if label_names is None:
        label_names = [f"Label_{i}" for i in range(y_pred.shape[1])]

    all_metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "ppv", "npv"]
    metrics_to_report = [m for m in metrics_to_report if m in all_metrics]

    results = {}
    table_data = []
    
    macro_metrics = {metric: [] for metric in metrics_to_report}
    macro_supports = []
    
    for i, label in enumerate(label_names):
        y_true_label = y_true[:, i]
        y_pred_binary = (y_pred[:, i] >= threshold).astype(int)
        support = int(np.sum(y_true_label))

        accuracy = accuracy_score(y_true_label, y_pred_binary)
        precision = precision_score(y_true_label, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_label, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_binary, zero_division=0)
        auc = roc_auc_score(y_true_label, y_pred[:, i])

        tn, fp, fn, tp = confusion_matrix(y_true_label, y_pred_binary).ravel()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        
        # Compute Confidence Intervals
        ci_accuracy = tuple(round(x, round_decimals) for x in exact_clopper_pearson(np.sum(y_true_label == y_pred_binary), len(y_true_label), alpha))
        ci_recall = tuple(round(x, round_decimals) for x in exact_clopper_pearson(tp, tp + fn, alpha)) if (tp + fn) > 0 else (np.nan, np.nan)
        ci_specificity = tuple(round(x, round_decimals) for x in exact_clopper_pearson(tn, tn + fp, alpha)) if (tn + fp) > 0 else (np.nan, np.nan)
        ci_precision = tuple(round(x, round_decimals) for x in bootstrap_ci(precision_score, y_true_label, y_pred[:, i], num_bootstraps, alpha, threshold))
        ci_f1 = tuple(round(x, round_decimals) for x in bootstrap_ci(f1_score, y_true_label, y_pred[:, i], num_bootstraps, alpha, threshold))
        ci_auc = tuple(round(x, round_decimals) for x in delong_auc_ci(y_true_label, y_pred[:, i], alpha))
        ci_ppv = tuple(round(x, round_decimals) for x in bootstrap_ci(lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0), y_true_label, y_pred[:, i], num_bootstraps, alpha, threshold))

        def npv_fn(y_t, y_p):
            cm = confusion_matrix(y_t, y_p)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                return tn / (tn + fn) if (tn + fn) > 0 else np.nan
            return np.nan
        ci_npv = tuple(round(x, round_decimals) for x in bootstrap_ci(npv_fn, y_true_label, y_pred[:, i], num_bootstraps, alpha, threshold))
        
        metrics_dict = {
            'accuracy': (round(accuracy, round_decimals), ci_accuracy),
            'precision': (round(precision, round_decimals), ci_precision),
            'recall': (round(recall, round_decimals), ci_recall),
            'specificity': (round(specificity, round_decimals), ci_specificity),
            'f1': (round(f1, round_decimals), ci_f1),
            'auc': (round(auc, round_decimals), ci_auc),
            'ppv': (round(ppv, round_decimals), ci_ppv),
            'npv': (round(npv, round_decimals), ci_npv),
            'support': support
        }
        results[label] = metrics_dict
        
        for metric in metrics_to_report:
            if metric in metrics_dict:
                macro_metrics[metric].append(metrics_dict[metric][0])
        macro_supports.append(support)

        row = [label] + [f"{metrics_dict[m][0]} ({metrics_dict[m][1][0]}, {metrics_dict[m][1][1]})" for m in metrics_to_report] + [support]
        table_data.append(row)
    
    # Macro-average metrics
    macro_metrics_avg = {metric: round(np.nanmean(macro_metrics[metric]), round_decimals) for metric in metrics_to_report}
    macro_support = int(np.mean(macro_supports))
    macro_metrics_avg["support"] = macro_support
    results['macro_avg'] = macro_metrics_avg

    # Micro-average metrics
    micro_y_true = y_true.ravel()
    micro_y_pred_binary = (y_pred.ravel() >= threshold).astype(int)
    micro_metrics = {
        'accuracy': round(accuracy_score(micro_y_true, micro_y_pred_binary), round_decimals),
        'support': int(np.sum(micro_y_true))
    }
    for metric in metrics_to_report:
        if metric in ["precision", "recall", "f1", "auc", "specificity"]:
            if metric == "precision":
                micro_metrics[metric] = round(precision_score(micro_y_true, micro_y_pred_binary, zero_division=0), round_decimals)
            elif metric == "recall":
                micro_metrics[metric] = round(recall_score(micro_y_true, micro_y_pred_binary, zero_division=0), round_decimals)
            elif metric == "f1":
                micro_metrics[metric] = round(f1_score(micro_y_true, micro_y_pred_binary, zero_division=0), round_decimals)
            elif metric == "auc":
                micro_metrics[metric] = round(roc_auc_score(micro_y_true, y_pred.ravel()), round_decimals)
            elif metric == "specificity":
                spec_list = []
                for i in range(y_pred.shape[1]):
                    yt = y_true[:, i]
                    yp = (y_pred[:, i] >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
                    spec_list.append(tn / (tn + fp) if (tn+fp) > 0 else np.nan)
                micro_metrics[metric] = round(np.nanmean(spec_list), round_decimals)
    results['micro_avg'] = micro_metrics

    table_data.append(["Macro Avg"] + [f"{macro_metrics_avg[m]}" for m in metrics_to_report] + [macro_support])
    table_data.append(["Micro Avg"] + [f"{micro_metrics[m]}" for m in metrics_to_report] + [micro_metrics['support']])

    if display_as_table:
        columns = ["Label"] + metrics_to_report + ["support"]
        df_results = pd.DataFrame(table_data, columns=columns)
        print(df_results)

    return results



# Compare ACC
def compare_accuracy_mcnemar(y_true, y_pred_1, y_pred_2, label_names=None, threshold=0.5, round_decimals=4):
    '''
    Compare model accuracy per label using McNemar's test.
    
    Args:
        y_true (np.ndarray, shape: n_samples, n_classes): Binary ground truth matrix.
        y_pred_1, y_pred_2 (np.ndarray, shape: n_samples, n_classes): Continuous predictions from two models.
        label_names (list, optional): Class labels.
        threshold (float): Threshold to binarize predictions.
        round_decimals (int): Decimal precision for p-value.
    
    Returns:
        dict: For each label, includes p-value, better model, and contingency table.
    '''

    num_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(num_labels)]
    
    results = {}
    for i, label in enumerate(label_names):
        y_true_label = y_true[:, i]
        y_pred1_binary = (y_pred_1[:, i] >= threshold).astype(int)
        y_pred2_binary = (y_pred_2[:, i] >= threshold).astype(int)
        
        # Build the 2x2 contingency table for discordant pairs
        contingency_table = np.zeros((2, 2))
        for yt, p1, p2 in zip(y_true_label, y_pred1_binary, y_pred2_binary):
            if p1 == yt and p2 != yt:
                contingency_table[0, 1] += 1
            elif p1 != yt and p2 == yt:
                contingency_table[1, 0] += 1
        
        test_result = mcnemar(contingency_table, exact=True)
        p_value = test_result.pvalue
        
        b = contingency_table[0, 1]
        c = contingency_table[1, 0]
        if b > c:
            better_model = "Model 1"
        elif c > b:
            better_model = "Model 2"
        else:
            better_model = "Tie"
        
        results[label] = {
            "p_value": round(p_value, round_decimals),
            "better_model": better_model,
            "contingency_table": contingency_table
        }
    return results


# Compare Recall
def compare_recall_mcnemar(y_true, y_pred_1, y_pred_2, label_names=None, threshold=0.5, round_decimals=4):
    '''
    Compare model recall (sensitivity) per label using McNemar's test.
    
    Args:
        y_true (np.ndarray, shape: n_samples, n_classes): Binary ground truth matrix.
        y_pred_1, y_pred_2 (np.ndarray, shape: n_samples, n_classes): Continuous predictions from two models.
        label_names (list, optional): Class labels.
        threshold (float): Threshold to binarize predictions.
        round_decimals (int): Decimal precision for p-value.
    
    Returns:
        dict: For each label, includes p-value, better model, and contingency table.
    '''

    num_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(num_labels)]
        
    results = {}
    for i, label in enumerate(label_names):
        # Only consider positive samples (for recall)
        pos_indices = np.where(y_true[:, i] == 1)[0]
        if len(pos_indices) == 0:
            results[label] = {"p_value": np.nan, "better_model": "N/A", "contingency_table": None}
            continue

        # Threshold continuous predictions
        y_true_label = y_true[pos_indices, i]
        y_pred1_binary = (y_pred_1[pos_indices, i] >= threshold).astype(int)
        y_pred2_binary = (y_pred_2[pos_indices, i] >= threshold).astype(int)
        
        # For recall (sensitivity): correct prediction is 1.
        contingency_table = np.zeros((2, 2))
        for yt, p1, p2 in zip(y_true_label, y_pred1_binary, y_pred2_binary):
            if p1 == 1 and p2 == 0:
                contingency_table[0, 1] += 1
            elif p1 == 0 and p2 == 1:
                contingency_table[1, 0] += 1
        
        test_result = mcnemar(contingency_table, exact=True)
        p_value = test_result.pvalue
        
        # Decide which model is better based on discordant pairs:
        b = contingency_table[0, 1]  # Model 1 correct, Model 2 wrong
        c = contingency_table[1, 0]  # Model 1 wrong, Model 2 correct
        if b > c:
            better_model = "Model 1"
        elif c > b:
            better_model = "Model 2"
        else:
            better_model = "Tie"
        
        results[label] = {
            "p_value": round(p_value, round_decimals),
            "better_model": better_model,
            "contingency_table": contingency_table
        }
    return results


# Compare Specificity
def compare_specificity_mcnemar(y_true, y_pred_1, y_pred_2, label_names=None, threshold=0.5, round_decimals=4):
    '''
    Compare model specificity per label using McNemar's test.
    
    Args:
        y_true (np.ndarray, shape: n_samples, n_classes): Binary ground truth matrix.
        y_pred_1, y_pred_2 (np.ndarray, shape: n_samples, n_classes): Continuous predictions from two models.
        label_names (list, optional): Class labels.
        threshold (float): Threshold to binarize predictions.
        round_decimals (int): Decimal precision for p-value.
    
    Returns:
        dict: For each label, includes p-value, better model, and contingency table.
    '''

    num_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(num_labels)]
        
    results = {}
    for i, label in enumerate(label_names):
        # Only consider negative samples (for specificity)
        neg_indices = np.where(y_true[:, i] == 0)[0]
        if len(neg_indices) == 0:
            results[label] = {"p_value": np.nan, "better_model": "N/A", "contingency_table": None}
            continue

        # Threshold continuous predictions
        y_true_label = y_true[neg_indices, i]
        y_pred1_binary = (y_pred_1[neg_indices, i] >= threshold).astype(int)
        y_pred2_binary = (y_pred_2[neg_indices, i] >= threshold).astype(int)
        
        # For specificity: correct prediction is 0.
        # Thus, if a model predicts 0 when the true label is 0, it is correct.
        contingency_table = np.zeros((2, 2))
        for yt, p1, p2 in zip(y_true_label, y_pred1_binary, y_pred2_binary):
            # p1 correct means p1==0, p2 incorrect means p2==1, etc.
            if p1 == 0 and p2 == 1:
                contingency_table[0, 1] += 1
            elif p1 == 1 and p2 == 0:
                contingency_table[1, 0] += 1
        
        test_result = mcnemar(contingency_table, exact=True)
        p_value = test_result.pvalue

        b = contingency_table[0, 1]  # Model 1 correct, Model 2 wrong (for negatives: Model 1 predicted 0 and Model 2 predicted 1)
        c = contingency_table[1, 0]  # Model 1 wrong, Model 2 correct
        if b > c:
            better_model = "Model 1"
        elif c > b:
            better_model = "Model 2"
        else:
            better_model = "Tie"
        
        results[label] = {
            "p_value": round(p_value, round_decimals),
            "better_model": better_model,
            "contingency_table": contingency_table
        }
    return results


# Bootstrap AUC
def bootstrap_auc_test(y_true, y_pred_1, y_pred_2, n_bootstraps=1000, seed=None):
    '''
    Perform bootstrap hypothesis testing for difference in AUC between two models.
    
    Args:
        y_true (np.ndarray, shape: n_samples, n_classes): Ground truth binary labels.
        y_pred_1, y_pred_2 (np.ndarray, shape: n_samples, n_classes): Continuous predictions for each model.
        n_bootstraps (int): Number of bootstrap resamples.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        tuple: (auc_diff, p_value) for the test.
    '''

    rng = np.random.RandomState(seed)
    auc1 = roc_auc_score(y_true, y_pred_1)
    auc2 = roc_auc_score(y_true, y_pred_2)
    auc_diff = auc1 - auc2
    
    boot_diffs = []
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        y_true_bs = y_true[indices]
        if len(np.unique(y_true_bs)) < 2:
            continue
        auc1_bs = roc_auc_score(y_true_bs, y_pred_1[indices])
        auc2_bs = roc_auc_score(y_true_bs, y_pred_2[indices])
        boot_diffs.append(auc1_bs - auc2_bs)
    boot_diffs = np.array(boot_diffs)
    if auc_diff > 0:
        p_value = np.mean(boot_diffs < 0)
    else:
        p_value = np.mean(boot_diffs > 0)
    p_value *= 2
    return auc_diff, p_value


# Compare AUCs
def compare_auc_bootstrap(y_true, y_pred_1, y_pred_2, label_names=None,
                          n_bootstraps=1000, round_decimals=4, seed=None):
    '''
    Compare AUCs of two models per label using bootstrapping.
    
    Args:
        y_true (np.ndarray, shape: n_samples, n_classes): Binary ground truth matrix (n_samples, n_classes).
        y_pred_1, y_pred_2 (np.ndarray, shape: n_samples, n_classes): Continuous prediction matrices from each model.
        label_names (list, optional): Class labels.
        n_bootstraps (int): Number of bootstrap iterations.
        round_decimals (int): Decimal precision.
        seed (int, optional): Random seed.
    
    Returns:
        dict: For each label, includes AUCs, difference, p-value, and better model.
    '''

    num_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(num_labels)]
    
    results = {}
    for i, label in enumerate(label_names):
        y_true_label = y_true[:, i]
        scores1 = y_pred_1[:, i]
        scores2 = y_pred_2[:, i]
        
        auc1 = roc_auc_score(y_true_label, scores1)
        auc2 = roc_auc_score(y_true_label, scores2)
        auc_diff, p_value = bootstrap_auc_test(y_true_label, scores1, scores2, n_bootstraps, seed)
        
        if auc_diff > 0:
            better_model = "Model 1"
        elif auc_diff < 0:
            better_model = "Model 2"
        else:
            better_model = "Tie"
        
        results[label] = {
            "auc_model1": round(auc1, round_decimals),
            "auc_model2": round(auc2, round_decimals),
            "auc_diff": round(auc_diff, round_decimals),
            "p_value": round(p_value, round_decimals),
            "better_model": better_model
        }
    return results
