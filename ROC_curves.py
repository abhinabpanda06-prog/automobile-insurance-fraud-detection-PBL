# =======================================
# ROC and Precision-Recall curves
# =======================================
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

strategies = ['none', 'class_weight', 'smote']
models = ['Logistic Regression', 'Random Forest', 'XGBoost']

# ROC
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, strategy in zip(axes, strategies):
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, fitted[(strategy, name)]['y_proba'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_title(strategy); ax.set_xlabel('False Positive Rate'); ax.legend(fontsize=8)
axes[0].set_ylabel('True Positive Rate')
plt.tight_layout(); plt.show()

# Precision-Recall
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, strategy in zip(axes, strategies):
    for name in models:
        yp = fitted[(strategy, name)]['y_proba']
        prec, rec, _ = precision_recall_curve(y_test, yp)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, yp):.3f})")
    ax.axhline(y_test.mean(), color='k', linestyle='--', linewidth=0.8)  # no-skill baseline
    ax.set_title(strategy); ax.set_xlabel('Recall'); ax.legend(fontsize=8)
axes[0].set_ylabel('Precision')
plt.tight_layout(); plt.show()
