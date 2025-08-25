import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score
)

RANDOM_STATE = 42

SCALER_MODE = "quantile"

def make_numeric_scaler(mode: str):
    mode = mode.lower()
    if mode == "standard":
        return StandardScaler()
    if mode == "minmax":
        return MinMaxScaler()
    if mode == "robust":
        return RobustScaler(quantile_range=(25.0, 75.0))
    if mode == "quantile":
        return QuantileTransformer(output_distribution="normal", subsample=200000, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown SCALER_MODE: {mode}")

def create_downsampled_classifier():
    print("="*70)
    print("SIMPLE DOWNSAMPLED INCOME CLASSIFIER")
    print("="*70)

    df = pd.read_csv("census_data.csv")
    print(f"Loaded {df.shape[0]:,} samples with {df.shape[1]} columns")

    # binary target from 'label'
    df['target'] = (~df['label'].astype(str).str.contains(r"-\s*50000", na=False)).astype(int)
    target_dist = df['target'].value_counts()
    print("Original distribution:")
    print(f"  - ≤$50K: {target_dist.get(0,0):,} ({target_dist.get(0,0)/len(df)*100:.1f}%)")
    print(f"  - >$50K: {target_dist.get(1,0):,} ({target_dist.get(1,0)/len(df)*100:.1f}%)")
    if target_dist.get(1,0) > 0:
        print(f"  - Imbalance ratio: {target_dist.get(0,0)/target_dist.get(1,1):.1f}:1")

    # Replace '?' -> NaN
    df = df.replace('?', np.nan)
    obj_cols = df.select_dtypes(include='object').columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        
    important_features = [
        'age', 'education', 'marital stat', 'sex', 'race',
        'weeks worked in year', 'capital gains', 'capital losses',
        'class of worker', 'major occupation code', 'num persons worked for employer'
    ]
    

    available_features = [f for f in important_features if f in df.columns]
    if not available_features:
        raise RuntimeError("None of the expected features were found in the dataset.")
    print(f"Selected {len(available_features)} key features")

    df_sel = df[available_features + ['target']].copy()

    
    # missing values handling (pre-fill numerics)
    categorical_cols = df_sel.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = [c for c in df_sel.select_dtypes(include=[np.number]).columns if c != 'target']

    for c in numerical_cols:
        df_sel[c] = pd.to_numeric(df_sel[c], errors='coerce')
        if c in ['capital gains', 'capital losses']:
            df_sel[c] = df_sel[c].fillna(0)
        else:
            pass

    print("Basic missing handling (final imputation will be in ColumnTransformer)")


    print("\nOUTLIER HANDLING (MODE-BASED):")
    initial_count = len(df_sel)
    total_outliers_handled = 0
    
    # handle age outliers (age == 0) by replacing with mode
    # rational: the over 50K is only about 12k lines of data, if remove outlier it will remove abt 1~2k of data in target_minority
    age_outliers = (df_sel['age'] == 0)
    if age_outliers.any():
        age_mode = df_sel[df_sel['age'] > 0]['age'].mode().iloc[0] if len(df_sel[df_sel['age'] > 0]) > 0 else df_sel['age'].median()
        outlier_count = age_outliers.sum()
        df_sel.loc[age_outliers, 'age'] = age_mode
        total_outliers_handled += outlier_count
        print(f"  - Replaced {outlier_count:,} invalid age values (age=0) with mode: {age_mode}")
    
    # same reason as above
    for col in ['capital gains', 'capital losses']:
        if col in df_sel.columns:
            q99 = df_sel[col].quantile(0.99)
            extreme_mask = df_sel[col] > q99
            if extreme_mask.any():
                # Calculate mode from non-extreme values
                non_extreme_values = df_sel[~extreme_mask][col]
                if len(non_extreme_values) > 0:
                    col_mode = non_extreme_values.mode().iloc[0] if len(non_extreme_values.mode()) > 0 else non_extreme_values.median()
                else:
                    col_mode = 0  # fallback
                
                outlier_count = extreme_mask.sum()
                df_sel.loc[extreme_mask, col] = col_mode
                total_outliers_handled += outlier_count
                print(f"  - Replaced {outlier_count:,} extreme {col} values (>{q99:,.0f}) with mode: {col_mode}")
                
    print(f"Outlier handling complete: {total_outliers_handled:,} outliers replaced with mode values")
    print(f"All samples retained: {len(df_sel):,} (no samples removed)")

    # Downsample majority to 1:1 for a simple, balanced baseline
    print("\nDOWNSAMPLING FOR CLASS BALANCE:")
    df_majority = df_sel[df_sel['target'] == 0]
    df_minority = df_sel[df_sel['target'] == 1]
    print(f" - Majority (≤$50K): {len(df_majority):,}")
    print(f" - Minority (>$50K): {len(df_minority):,}")

    target_majority_size = min(len(df_majority), len(df_minority))
    df_majority_down = resample(
        df_majority,
        replace=False,
        n_samples=target_majority_size,
        random_state=RANDOM_STATE
    )
    df_balanced = pd.concat([df_majority_down, df_minority], axis=0)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    bal_dist = df_balanced['target'].value_counts()
    print("After downsampling:")
    print(f" - ≤$50K: {bal_dist.get(0,0):,}")
    print(f" - >$50K: {bal_dist.get(1,0):,}")
    if bal_dist.get(1,0) > 0:
        print(f"  - New ratio: {bal_dist.get(0,0)/bal_dist.get(1,1):.1f}:1")
    print(f"  - Total samples: {len(df_balanced):,} (from {len(df_sel):,})")

    # Train/test split
    X = df_balanced.drop(columns=['target'])
    y = df_balanced['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    print("\nData split:")
    print(f"  - Training: {X_train.shape}")
    print(f"  - Testing : {X_test.shape}")

    numeric_cols_now = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols_now = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    scaler = make_numeric_scaler(SCALER_MODE)

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.01,
            sparse_output=False
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols_now),
            ("cat", cat_pipeline, categorical_cols_now),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )

    # pipline
    lr_pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])

    rf_pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

    xgb_pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        ))
    ])

    results = {}

    # Logistic Regression
    print(f"\nTraining Logistic Regression (scaler = {SCALER_MODE}) ...")
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = {
        'model': lr_pipe,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_proba_lr),
        'pr_auc': average_precision_score(y_test, y_proba_lr),
        'y_pred': y_pred_lr,
        'y_proba': y_proba_lr
    }
    print(f" Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
    print(f" ROC-AUC : {results['Logistic Regression']['roc_auc']:.4f}")
    print(f" PR-AUC  : {results['Logistic Regression']['pr_auc']:.4f}")
    print("  Classification Report:")
    print("     " + classification_report(y_test, y_pred_lr, target_names=['≤$50K', '>$50K']).replace('\n','\n     '))
    cm = confusion_matrix(y_test, y_pred_lr)
    print("  Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,}, FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")

    # Random Forest
    print("\nTraining Random Forest (with same preprocessor) ...")
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {
        'model': rf_pipe,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_proba_rf),
        'pr_auc': average_precision_score(y_test, y_proba_rf),
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf
    }
    print(f" Accuracy: {results['Random Forest']['accuracy']:.4f}")
    print(f" ROC-AUC : {results['Random Forest']['roc_auc']:.4f}")
    print(f" PR-AUC  : {results['Random Forest']['pr_auc']:.4f}")
    print("  Classification Report:")
    print("     " + classification_report(y_test, y_pred_rf, target_names=['≤$50K', '>$50K']).replace('\n','\n     '))
    cm = confusion_matrix(y_test, y_pred_rf)
    print("  Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,}, FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")

    # XGBoost
    print("\nTraining XGBoost (with same preprocessor) ...")
    xgb_pipe.fit(X_train, y_train)
    y_pred_xgb = xgb_pipe.predict(X_test)
    y_proba_xgb = xgb_pipe.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'model': xgb_pipe,
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_proba_xgb),
        'pr_auc': average_precision_score(y_test, y_proba_xgb),
        'y_pred': y_pred_xgb,
        'y_proba': y_proba_xgb
    }
    print(f" Accuracy: {results['XGBoost']['accuracy']:.4f}")
    print(f" ROC-AUC : {results['XGBoost']['roc_auc']:.4f}")
    print(f" PR-AUC  : {results['XGBoost']['pr_auc']:.4f}")
    print("  Classification Report:")
    print("     " + classification_report(y_test, y_pred_xgb, target_names=['≤$50K', '>$50K']).replace('\n','\n     '))
    cm = confusion_matrix(y_test, y_pred_xgb)
    print("  Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,}, FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")

    # Feature importance analysis
    feature_names = rf_pipe.named_steps["prep"].get_feature_names_out()
    
    # RF Feature importance
    rf_importances = rf_pipe.named_steps["clf"].feature_importances_
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importances
    }).sort_values('importance', ascending=False)
    
    # XGBoost Feature importance
    xgb_importances = xgb_pipe.named_steps["clf"].feature_importances_
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_importances
    }).sort_values('importance', ascending=False)
    
    # Pick best by ROC-AUC
    best_name = max(results.keys(), key=lambda k: results[k]['pr_auc'])
    best = results[best_name]

    print("\n" + "="*70)
    print("DOWNSAMPLED CLASSIFIER RESULTS")
    print("="*70)
    print(f"\nBEST MODEL: {best_name}")
    print(f" Accuracy: {best['accuracy']:.4f}")
    print(f" ROC-AUC : {best['roc_auc']:.4f}")
    print(f" PR-AUC  : {best['pr_auc']:.4f}")

    # Save best model + artifacts
    model_package = {
        'model_name': best_name,
        'model': best['model'],          
        'selected_features': available_features,
        'performance': {
            'accuracy': best['accuracy'],
            'roc_auc': best['roc_auc'],
            'pr_auc': best['pr_auc']
        },
        'scaler_mode': SCALER_MODE,
        'feature_names_out': feature_names.tolist()
    }
    with open("simple_downsampled_model.pkl", "wb") as f:
        pickle.dump(model_package, f)
    return results, model_package, rf_importance, xgb_importance, df_balanced




if __name__ == "__main__":
    results, package, rf_importance, xgb_importance, balanced_df = create_downsampled_classifier()



