# =====================
# CONFUSION MATRICES
# =====================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

important = [
    ('none', 'XGBoost'),                       # baseline, no imbalance handling
    ('class_weight', 'XGBoost'),                # best overall F1
    ('class_weight', 'Random Forest'),          # stays near-blind to fraud even reweighted
    ('class_weight', 'Logistic Regression'),    # strongest rebalancing response
    ('smote', 'Logistic Regression'),           # SMOTE vs. class-weight, same model
]

fig, axes = plt.subplots(1, len(important), figsize=(4.2*len(important), 4.2))
for ax, (strategy, name) in zip(axes, important):
    cm = confusion_matrix(y_test, fitted[(strategy, name)]['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=['Legit', 'Fraud']).plot(
        ax=ax, colorbar=False, cmap='Blues', values_format='d'
    )
    ax.set_title(f"{name}\n({strategy})", fontsize=10)
plt.tight_layout()
plt.show()
