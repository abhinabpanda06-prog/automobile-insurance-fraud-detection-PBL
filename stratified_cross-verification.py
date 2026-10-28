# ============================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION — resampling inside training folds only
# ============================================================
import time
import warnings
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, average_precision_score)

# Expected: a rare category (e.g. a Make) can be absent from a training fold; handle_unknown='ignore' covers it
warnings.filterwarnings('ignore', message='Found unknown categories')

# CV runs on the training split only — X_test / y_test stay untouched as the final hold-out.
# (To cross-validate on the whole dataset instead: X_cv, y_cv = X, y)
X_cv, y_cv = X_train, y_train

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
start = time.time()

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), start=1):
    X_tr, X_va = X_cv.iloc[tr_idx], X_cv.iloc[va_idx]
    y_tr, y_va = y_cv.iloc[tr_idx], y_cv.iloc[va_idx]

    # Preprocessing (median, scaler mean/std, one-hot vocabulary) is fit on THIS training fold only
    prep = clone(preprocessor)
    X_tr_p = prep.fit_transform(X_tr, y_tr)
    X_va_p = prep.transform(X_va)

    neg_f, pos_f = np.bincount(y_tr)  # class ratio recomputed inside the training fold (XGBoost scale_pos_weight)

    for strategy in ['none', 'class_weight', 'smote']:
        if strategy == 'smote':
            # === SMOTE applied HERE ONLY: this training fold, inside the CV loop ===
            # The validation fold (X_va_p, y_va) is never resampled and keeps the native ~6% fraud rate
            X_fit, y_fit = SMOTE(random_state=42).fit_resample(X_tr_p, y_tr)
            print(f"[fold {fold}] SMOTE on training fold: {np.bincount(y_tr)} -> {np.bincount(y_fit)} "
                  f"| validation fold untouched: {np.bincount(y_va)}")
        else:
            X_fit, y_fit = X_tr_p, y_tr

        models = get_models(strategy)
        if strategy == 'class_weight':
            models['XGBoost'].set_params(scale_pos_weight=neg_f / pos_f)

        for name, model in models.items():
            model.fit(X_fit, y_fit)
            y_pred = model.predict(X_va_p)
            y_proba = model.predict_proba(X_va_p)[:, 1]
            cv_rows.append({
                'Fold': fold, 'Strategy': strategy, 'Model': name,
                'Precision': precision_score(y_va, y_pred, pos_label=1, zero_division=0),
                'Recall': recall_score(y_va, y_pred, pos_label=1, zero_division=0),
                'F1': f1_score(y_va, y_pred, pos_label=1, zero_division=0),
                'Accuracy': accuracy_score(y_va, y_pred),
                'ROC-AUC': roc_auc_score(y_va, y_proba),
                'PR-AUC (AP)': average_precision_score(y_va, y_proba),
            })

print(f"\nCV finished in {time.time() - start:.0f}s")
cv_df = pd.DataFrame(cv_rows)  # one row per fold x strategy x model (45 rows)

# --- Summary: mean ± std across the 5 folds, ranked by mean F1 ---
metrics = ['Precision', 'Recall', 'F1', 'Accuracy', 'ROC-AUC', 'PR-AUC (AP)']
agg = cv_df.groupby(['Strategy', 'Model'])[metrics].agg(['mean', 'std'])
agg = agg.sort_values(('F1', 'mean'), ascending=False)
cv_summary = pd.DataFrame({
    m: agg[(m, 'mean')].map('{:.3f}'.format) + ' ± ' + agg[(m, 'std')].map('{:.3f}'.format)
    for m in metrics
})
display(cv_summary)
