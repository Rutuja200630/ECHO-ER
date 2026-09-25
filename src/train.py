import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

def construct_training_data(candidates_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 16: Training Data Construction
    candidates_df: s1_id, s2_id, ... (features)
    gt_df: source1_entity_id, matched_entity_ids (comma separated)
    """
    df = candidates_df.copy()
    
    # Explode GT to get positive pairs
    gt_df['s2_id_list'] = gt_df['matched_entity_ids'].apply(
        lambda x: x.split(',') if pd.notna(x) and str(x).strip() else []
    )
    
    gt_exploded = gt_df[['source1_entity_id', 's2_id_list']].explode('s2_id_list')
    gt_exploded.columns = ['s1_id', 's2_id']
    gt_exploded['label'] = 1
    
    # Merge labels into candidates
    df = df.merge(gt_exploded, on=['s1_id', 's2_id'], how='left')
    df['label'] = df['label'].fillna(0).astype(int)
    
    return df

def train_lightgbm(df: pd.DataFrame, feature_cols: list, model_dir: str = '../output'):
    """
    Phase 17 & 18: Train/Validation Split & LightGBM
    Ensures leakage-safe validation using GroupKFold on s1_id.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    X = df[feature_cols]
    y = df['label']
    groups = df['s1_id']
    
    # Simple validation using 1 fold for early stopping
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X, y, groups))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss', # We will optimize F0.5 via threshold later
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    
    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    
    # Phase 19: Probability Calibration (optional, skipping for base model but structure exists)
    
    model_path = os.path.join(model_dir, 'lgb_model.txt')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return model

def get_hard_negatives(df: pd.DataFrame, model, feature_cols: list, threshold: float = 0.5) -> pd.DataFrame:
    """
    Phase 20: Hard Negative Mining
    Finds highly scored false positives to add to next training round.
    """
    preds = model.predict(df[feature_cols])
    df_pred = df.copy()
    df_pred['pred'] = preds
    
    # False positives
    hard_negs = df_pred[(df_pred['label'] == 0) & (df_pred['pred'] > threshold)]
    print(f"Found {len(hard_negs)} hard negatives.")
    return hard_negs
