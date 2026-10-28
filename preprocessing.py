# ============================================================
# PREPROCESSING
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv('dataset.csv', compression='zip', encoding='utf-8-sig')
TARGET = 'FraudFound_P'

# --- Deterministic data-quality fixes (safe pre-split: no fitted statistics involved) ---
df = df.drop(columns=['PolicyNumber'])  # ID column, no predictive meaning

make_fix = {'Mecedes': 'Mercedes', 'Nisson': 'Nissan', 'Porche': 'Porsche', 'Accura': 'Acura'}
df['Make'] = df['Make'].replace(make_fix)

corrupted = (df['DayOfWeekClaimed'] == '0') | (df['MonthClaimed'] == '0')
df = df[~corrupted].reset_index(drop=True)

df['Age'] = df['Age'].replace(0, np.nan)  # mark disguised-missing; impute after split, not before

# --- Split X / y ---
X = df.drop(columns=[TARGET])
y = df[TARGET]
numeric_cols = list(X.select_dtypes(include='number').columns)
categorical_cols = list(X.select_dtypes(include='object').columns)

# --- Train/test split BEFORE any fitted transform or resampling ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, fraud rate {y_train.mean():.4f}")
print(f"Test:  {X_test.shape}, fraud rate {y_test.mean():.4f}")

# --- Impute + scale numeric, one-hot encode categorical — fit on TRAIN, apply to both ---
numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
categorical_pipe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_cols),
    ('cat', categorical_pipe, categorical_cols)
])

X_train_proc = preprocessor.fit_transform(X_train, y_train)
X_test_proc = preprocessor.transform(X_test)
print("Processed feature matrix:", X_train_proc.shape)
