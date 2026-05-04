"""
fsr_effort_log*.csv 파일로 SVM 접촉 감지 모델 재학습.

사용법:
  python3 train_svm_model.py

출력:
  svm_collision_model.pkl  (같은 폴더)
"""

import csv
import os
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import resample

# contact_detector_svm.py 와 동일한 파라미터
WINDOW_SIZE           = 10
LOCAL_BASELINE_SAMPLES = 3
JOINT_NAMES           = ['joint1', 'joint2', 'joint3', 'joint4']

LOG_DIR  = Path(__file__).parent
LOG_FILES = sorted(LOG_DIR.glob('fsr_effort_log*.csv'))
OUT_PATH  = LOG_DIR / 'svm_collision_model.pkl'


def load_raw(path):
    """CSV → numpy array [v1..v4, e1..e4] + label"""
    samples, labels = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = [float(row[f'v{i}']) for i in range(1, 5)]
            e = [float(row[f'e{i}']) for i in range(1, 5)]
            samples.append(v + e)
            labels.append(int(row['label']))
    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int32)


def make_feature(history):
    """contact_detector_svm._make_svm_feature() 와 동일한 feature 생성."""
    prev   = history[0:1]          # (1, 8)
    window = history[1:]           # (10, 8)

    velocities   = window[:, :4]
    efforts      = window[:, 4:]
    baseline     = efforts[:LOCAL_BASELINE_SAMPLES].mean(axis=0, keepdims=True)
    effort_delta = efforts - baseline

    raw      = np.concatenate([velocities, effort_delta], axis=1)          # (10, 8)
    prev_raw = np.concatenate([prev[:, :4], prev[:, 4:] - baseline], axis=1)  # (1, 8)
    diff     = np.abs(raw - np.concatenate([prev_raw, raw[:-1]], axis=0))  # (10, 8)

    return np.concatenate([raw, diff], axis=1).reshape(-1)  # (160,)


def build_windows(samples, labels):
    """슬라이딩 윈도우로 feature/label 생성."""
    n = WINDOW_SIZE + 1  # 11개 (prev 1 + window 10)
    X, y = [], []
    for i in range(n, len(samples) + 1):
        history = samples[i - n: i]          # (11, 8)
        window_labels = labels[i - WINDOW_SIZE: i]  # 최근 10개 label

        feat  = make_feature(history)
        # 윈도우 내 label=1 비율 > 30% → 접촉 윈도우
        label = 1 if window_labels.mean() > 0.3 else 0
        X.append(feat)
        y.append(label)
    return np.array(X), np.array(y)


def main():
    if not LOG_FILES:
        print('로그 파일 없음:', LOG_DIR)
        return

    all_X, all_y = [], []
    for path in LOG_FILES:
        samples, labels = load_raw(path)
        X, y = build_windows(samples, labels)
        all_X.append(X)
        all_y.append(y)
        pos = y.sum()
        print(f'  {path.name}: {len(y)}개 윈도우 (TAP={pos}, 정상={len(y)-pos})')

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    print(f'\n전체: {len(y)}개  TAP={y.sum()}  정상={len(y)-y.sum()}')

    # 클래스 불균형 처리 — 정상 클래스 언더샘플링
    idx_pos  = np.where(y == 1)[0]
    idx_neg  = np.where(y == 0)[0]
    n_target = min(len(idx_pos) * 3, len(idx_neg))  # 정상:TAP = 3:1
    idx_neg_down = resample(idx_neg, n_samples=n_target, random_state=42, replace=False)
    idx_all  = np.concatenate([idx_pos, idx_neg_down])
    np.random.shuffle(idx_all)
    X_bal, y_bal = X[idx_all], y[idx_all]
    print(f'언더샘플링 후: TAP={y_bal.sum()}  정상={(y_bal==0).sum()}')

    # 학습
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=1.0, gamma='scale',
                       probability=True, class_weight='balanced')),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_bal, y_bal, cv=cv, scoring='f1')
    print(f'5-fold CV F1: {scores.mean():.3f} ± {scores.std():.3f}')

    model.fit(X_bal, y_bal)

    import joblib
    joblib.dump(model, OUT_PATH)
    print(f'\n✅ 모델 저장: {OUT_PATH}')


if __name__ == '__main__':
    main()
