# @author semese

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import scipy
import seaborn as sns
import shap
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

shap.initjs()

sns.set(font_scale=1.25)
sns.set_palette("tab20")
sns.set_style("ticks")


# ----------------------------------- Model Training and Evaluation -------------------------------------------------- #
def plot_loss_comparison(tr_loss, val_loss, figsize=(10, 5), ax=None):
    """
    Training and validation loss evolution plot over the epochs.

    :param list tr_loss: values of training loss in each epoch
    :param list val_loss: values of validation loss in each epoch
    :param tuple figsize: figure size; optional
    :param matplotlib.axes ax: axes that will contain the plot; optional
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(tr_loss, label="Training")
    ax.plot(val_loss, label="Validation")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    ax.legend()


def plot_evaluation(y_true, y_pred, y_score, figsize=(21, 7), label="", normalise=False):
    """
    Create a plot of the evaluation of predictions, containing the ROC, PRC and confusion matrix.

    :param np.ndarray y_true: ground truth (correct) labels of shape (n_samples,)
    :param np.ndarray y_pred: predicted labels, as returned by a classifier of shape (n_samples,)
    :param np.ndarray y_score: estimated probabilities or output of a decision function
    :param tuple figsize: figure size; optional
    :param str label: name of the classifier output; optional
    :param bool normalise: if True, the normalised confusion matrix is plotted
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    n_labels = len(np.unique(y_true))
    if n_labels > 2:
        plot_roc_multiclass(y_true, y_score, n_labels, label, ax=ax1)
        plot_prc_multiclass(y_true, y_score, n_labels, label, ax=ax2)
    else:
        plot_roc(y_true, y_score[:, 1], label, ax=ax1)
        plot_prc(y_true, y_score[:, 1], label, ax=ax2)
    _ = skplt.metrics.plot_confusion_matrix(y_true, y_pred, ax=ax3, cmap="Blues", normalize=normalise)
    fig.tight_layout()


def plot_roc(y_true, y_score, out_name, figsize=(10, 5), ax=None):
    """
    Plot Receiver operating characteristic (ROC) curve.

    :param np.ndarray y_true: true binary labels of shape (n_samples,)
    :param np.ndarray y_score: estimated probabilities or output of a decision function of shape (n_samples,)
    :param str out_name: label for the classification task
    :param tuple figsize: figure size; optional
    :param matplotlib.axes ax: axes that will contain the plot; optional
    """
    false_positive_rate, recall, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(false_positive_rate, recall)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(out_name)
    ax.plot(false_positive_rate, recall, label="AUC = %0.3f" % roc_auc)
    legend = ax.legend(loc="lower right", shadow=True, frameon=True)

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor("whitesmoke")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel("Sensitivity")
    ax.set_xlabel("1-Specificity")


def plot_roc_multiclass(
        y_true, y_score, n_classes, out_name, ax=None, figsize=(10, 5)
):
    """
    Plot Receiver operating characteristic (ROC) curve.

    :param np.ndarray y_true: true labels of shape (n_samples,)
    :param np.ndarray y_score: estimated probabilities or output of a decision function of shape (n_samples, n_classes)
    :param int n_classes: number of possible labels
    :param str out_name: label for the classification task
    :param matplotlib.axes ax: axes that will contain the plot; optional
    :param tuple figsize: figure size; optional
    """
    y_true = label_binarize(y_true, classes=np.arange(n_classes))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(out_name)

    for i in range(n_classes):
        ax.plot(
            fpr[i],
            tpr[i],
            label="ROC - class {0} (AUC = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-avg ROC (AUC = {0:0.2f})" "".format(roc_auc["micro"]),

        linewidth=2
    )

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1-Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", shadow=True, facecolor="whitesmoke", frameon=True)


def plot_prc(y_true, y_score, out_name, figsize=(10, 5), ax=None):
    """
    Plot Precision-Recall curve.

    :param np.ndarray y_true: true binary labels of shape (n_samples,)
    :param np.ndarray y_score: estimated probabilities or output of a decision function of shape (n_samples,)
    :param str out_name: label for the classification task
    :param tuple figsize: figure size; optional
    :param matplotlib.axes ax: axes that will contain the plot; optional
    """

    # compute precision-recall pairs for different probability thresholds
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(out_name)
    ax.plot(recall, precision, label="AUC =  %0.3f" % average_precision)
    ax.legend(loc="lower right", shadow=True, facecolor="whitesmoke", frameon=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")


def plot_prc_multiclass(
        y_true, y_score, n_classes, out_name, ax=None, figsize=(10, 5)
):
    """
    Precision-Recall plot of a multi-class problem.

    :param np.ndarray y_true: true labels of shape (n_samples,)
    :param np.ndarray y_score: estimated probabilities or output of a decision function of shape (n_samples, n_classes)
    :param int n_classes: number of possible labels
    :param str out_name: label for the classification task
    :param matplotlib.axes ax: axes that will contain the plot; optional
    :param tuple figsize: figure size; optional
    """
    y_true = label_binarize(y_true, classes=np.arange(n_classes))
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true, y_score, average="micro"
    )

    # Plot PRC curve for each class and iso-f1 curves
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")

    for i in range(n_classes):
        l, = ax.plot(recall[i], precision[i])
        lines.append(l)
        labels.append(
            "PRC for class {0} (AUC =  {1:0.2f})" "".format(i, average_precision[i])
        )

    l, = ax.plot(recall["micro"], precision["micro"], linewidth=2)
    lines.append(l)
    labels.append(
        "micro-avg PRC (AUC =  {0:0.2f})" "".format(average_precision["micro"])
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(
        lines,
        labels,
        loc="lower right",
        shadow=True,
        facecolor="whitesmoke",
        frameon=True,
    )
    ax.set_title(out_name)


# ---------------------------------- Classifier Feature Importance --------------------------------------------------- #
def create_feature_importance_plot(df_train, df_test, clf, columns, n_exp):
    """
    Compute SHAP values and create feature importance plot of a trained classifier,
    given the training and test sets.

    :param pd.DataFrame df_train: data frame containing the training data
    :param pd.DataFrame df_test: data frame containing the test data
    :param clf: train sklearn classifier
    :param list columns: list of column labels to consider from the dataframe
    :param int n_exp: number of samples to use for SHAP value computation
    """
    # compute k-means on training set to reduce the computation burden
    X_train_kmeans = shap.kmeans(df_train[columns], 100)

    # explain some the predictions in the test set
    explainer = shap.KernelExplainer(clf.predict_proba, X_train_kmeans)
    shap_values = explainer.shap_values(df_test[columns].iloc[:n_exp, :], l1_reg="aic")

    # create shap summary plot
    shap.summary_plot(shap_values, df_test[columns].iloc[:n_exp, :], color=plt.get_cmap("tab20"));
