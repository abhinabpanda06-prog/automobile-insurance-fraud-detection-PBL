# ============================================================
# BASELINE MODELS — 3 models x 3 imbalance strategies
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

def get_models(strategy):
    if strategy == 'class_weight':
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
        }
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }

fitted = {}  # (strategy, model_name) -> fitted model + predictions, reused by the cells below
results = []

for strategy in ['none', 'class_weight', 'smote']:
    if strategy == 'smote':
        # === SMOTE applied HERE ONLY: training fold, after the train/test split, inside this branch ===
        # X_test_proc / y_test are never touched — test keeps the native ~6% fraud rate throughout
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_train_proc, y_train)
        print(f"[SMOTE] training fold rebalanced: {np.bincount(y_train)} -> {np.bincount(y_tr)}")
    else:
        X_tr, y_tr = X_train_proc, y_train

    for name, model in get_models(strategy).items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]

        fitted[(strategy, name)] = {'y_pred': y_pred, 'y_proba': y_proba}
        results.append({
            'Strategy': strategy, 'Model': name,
            'Precision (fraud)': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'Recall (fraud)': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'F1 (fraud)': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        })

results_df = pd.DataFrame(results).sort_values(['Strategy', 'F1 (fraud)'], ascending=[True, False])
print(results_df.round(4).to_string(index=False))
