
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings(action='ignore')

train_df = pd.read_csv('./train.csv')
val_df = pd.read_csv('./val.csv')
test_df = pd.read_csv('./test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# Validation set 사기 거래 비율 확인
val_normal, val_fraud = val_df['Class'].value_counts()
val_contamination = val_fraud / (val_normal + val_fraud)
print(f'Validation contamination ratio: {val_contamination:.6f}')
print(f'Normal: {val_normal}, Fraud: {val_fraud}')

"""## Semi-Supervised Learning 데이터 준비"""

# Train 데이터 (라벨 없음)
train_x = train_df.drop(columns=['ID']).copy()

# Validation 데이터 (라벨 있음)
val_x = val_df.drop(columns=['ID', 'Class']).copy()
val_y = val_df['Class'].copy()

# Semi-supervised를 위해 일부 validation 데이터를 train에 추가
# Validation을 train/test로 분할 (일부는 학습에 사용, 일부는 평가에 사용)
val_train_x, val_test_x, val_train_y, val_test_y = train_test_split(
    val_x, val_y, test_size=0.2, random_state=529, stratify=val_y
)

# Oversampling 적용
print("\n=== Oversampling 적용 ===")

# 라벨된 데이터에만 oversampling 적용
print(f"Before oversampling: {np.bincount(val_train_y)}")

# SMOTETomek
selected_method = 'SMOTETomek'
smote_method = SMOTETomek(
        smote=SMOTE(sampling_strategy=0.3, random_state=529),
        tomek=None,
        random_state=529
    )
val_train_x_balanced, val_train_y_balanced = smote_method.fit_resample(val_train_x, val_train_y)

print(f"After {selected_method}: {np.bincount(val_train_y_balanced)}")
print(f"New contamination ratio: {val_train_y_balanced.sum() / len(val_train_y_balanced):.6f}")

# Semi-supervised 학습용 데이터 생성 (밸런싱된 라벨 데이터 사용)
semi_train_x = pd.concat([train_x, pd.DataFrame(val_train_x_balanced, columns=val_train_x.columns)], ignore_index=True)
semi_train_y = pd.concat([
    pd.Series([-1] * len(train_x)),  # unlabeled
    pd.Series(val_train_y_balanced)  # balanced labeled data
], ignore_index=True)

# 데이터 표준화
scaler = StandardScaler()
semi_train_x_scaled = scaler.fit_transform(semi_train_x)
val_test_x_scaled = scaler.transform(val_test_x)

# Label Propagation
print("\n=== Label Propagation 적용 ===")

# Label Propagation을 사용하여 unlabeled 데이터에 pseudo-label 생성
label_prop = LabelPropagation(**{'kernel': 'knn', 'n_neighbors': 7, 'gamma': 20},)

# 학습
label_prop.fit(semi_train_x_scaled, semi_train_y)

# Pseudo-labels 생성
pseudo_labels = label_prop.predict(semi_train_x_scaled)
pseudo_proba = label_prop.predict_proba(semi_train_x_scaled)

# Confidence threshold 지정
selected_threshold = 0.6  # 직접 임의로 지정한 값

high_confidence_mask = np.max(pseudo_proba, axis=1) >= selected_threshold

print(f"Confidence threshold: {selected_threshold}")
print(f"High confidence pseudo-labels: {sum(high_confidence_mask)}/{len(pseudo_labels)}")

# 높은 confidence를 가진 데이터만 사용하여 학습
confident_x = semi_train_x_scaled[high_confidence_mask]
confident_y = pseudo_labels[high_confidence_mask]

# Normal과 Fraud 비율 확인
normal_count = sum(confident_y == 0)
fraud_count = sum(confident_y == 1)
confident_contamination = fraud_count / (normal_count + fraud_count) if (normal_count + fraud_count) > 0 else val_contamination

print(f"Confident pseudo-labels - Normal: {normal_count}, Fraud: {fraud_count}")
print(f"Confident contamination ratio: {confident_contamination:.6f}")

# Isolation Forest 모델 학습
normal_data = confident_x[confident_y == 0]

model_semi = IsolationForest(
    n_estimators=300,
    max_samples='auto',
    contamination=confident_contamination,
    random_state=529,
    verbose=0
)
model_semi.fit(normal_data)

# 1. 기존 방식: 라벨 정보 없이 학습
model_baseline = IsolationForest(
    n_estimators=300,
    max_samples='auto',
    contamination=val_contamination * 5,
    random_state=529,
    verbose=0
)
model_baseline.fit(semi_train_x_scaled)

# 2. Oversampling만 적용한 모델 (Semi-supervised 없이)
model_oversampling_only = IsolationForest(
    n_estimators=300,
    max_samples='auto',
    contamination=confident_contamination,
    random_state=529,
    verbose=0
)
# 밸런싱된 정상 데이터만 사용
balanced_normal = val_train_x_balanced[val_train_y_balanced == 0]
balanced_normal_scaled = scaler.transform(balanced_normal)
model_oversampling_only.fit(balanced_normal_scaled)

# 3. Semi-supervised만 적용한 모델 (Oversampling 없이)
# 원본 라벨 데이터로 semi-supervised 학습
semi_train_x_original = pd.concat([train_x, val_train_x], ignore_index=True)
semi_train_y_original = pd.concat([
    pd.Series([-1] * len(train_x)),
    val_train_y
], ignore_index=True)

semi_train_x_original_scaled = scaler.fit_transform(semi_train_x_original)
label_prop_original = LabelPropagation(**{'kernel': 'knn', 'n_neighbors': 7, 'gamma': 20})
label_prop_original.fit(semi_train_x_original_scaled, semi_train_y_original)

pseudo_labels_original = label_prop_original.predict(semi_train_x_original_scaled)
pseudo_proba_original = label_prop_original.predict_proba(semi_train_x_original_scaled)
high_conf_mask_original = np.max(pseudo_proba_original, axis=1) >= selected_threshold

confident_x_original = semi_train_x_original_scaled[high_conf_mask_original]
confident_y_original = pseudo_labels_original[high_conf_mask_original]
normal_data_original = confident_x_original[confident_y_original == 0]

model_semi_only = IsolationForest(
    n_estimators=300,
    max_samples='auto',
    contamination=val_contamination * 5,
    random_state=42,
    verbose=0
)
model_semi_only.fit(normal_data_original)


def get_pred_label(model_pred):
    #IsolationForest 출력을 0/1 라벨로 변환
    model_pred = np.where(model_pred == 1, 0, model_pred)
    model_pred = np.where(model_pred == -1, 1, model_pred)
    return model_pred

def evaluate_model_detailed(model, X, y, model_name="Model"):
    pred = model.predict(X)
    pred = get_pred_label(pred)

    f1_macro = f1_score(y, pred, average='macro')
    f1_weighted = f1_score(y, pred, average='weighted')
    f1_binary = f1_score(y, pred, average='binary', zero_division=0)

    precision_macro = precision_score(y, pred, average='macro', zero_division=0)
    recall_macro = recall_score(y, pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y, pred)

    # 클래스별 상세 지표
    precision_class = precision_score(y, pred, average=None, zero_division=0)
    recall_class = recall_score(y, pred, average=None, zero_division=0)

    print(f"\n=== {model_name} Results ===")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (binary): {f1_binary:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    if len(precision_class) >= 2:
        print(f"Class 0 (Normal) - Precision: {precision_class[0]:.4f}, Recall: {recall_class[0]:.4f}")
        print(f"Class 1 (Fraud) - Precision: {precision_class[1]:.4f}, Recall: {recall_class[1]:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, pred, zero_division=0))

    return {
        'f1_macro': f1_macro,
        'f1_binary': f1_binary,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'accuracy': accuracy,
        'predictions': pred
    }


"""## Model Evaluation"""

print("\n" + "=" * 80)
print("모델 성능 비교 (4가지 접근법)")
print("=" * 80)

# 모든 모델 평가
combined_results = evaluate_model_detailed(model_semi, val_test_x_scaled, val_test_y, "Semi-Supervised + Oversampling")
oversampling_results = evaluate_model_detailed(model_oversampling_only, val_test_x_scaled, val_test_y,
                                               "Oversampling Only")
semi_only_results = evaluate_model_detailed(model_semi_only, val_test_x_scaled, val_test_y, "Semi-Supervised Only")
baseline_results = evaluate_model_detailed(model_baseline, val_test_x_scaled, val_test_y, "Baseline")

print("\n" + "=" * 100)
print("MODEL COMPARISON SUMMARY")
print("=" * 100)

models = {
    'Semi-Supervised + Oversampling': combined_results,
    'Oversampling Only': oversampling_results,
    'Semi-Supervised Only': semi_only_results,
    'Baseline': baseline_results
}

print(f"{'Model':<30} {'F1 (Binary)':<12} {'F1 (Macro)':<12} {'Precision':<12} {'Recall':<12} {'Accuracy':<12}")
print("-" * 100)

for name, results in models.items():
    print(f"{name:<30} {results['f1_binary']:<12.4f} {results['f1_macro']:<12.4f} "
          f"{results['precision_macro']:<12.4f} {results['recall_macro']:<12.4f} {results['accuracy']:<12.4f}")

# 개선도 계산
print(f"\n개선도 분석 (vs Baseline):")
print(f"Semi-Supervised + Oversampling: {combined_results['f1_binary'] - baseline_results['f1_binary']:+.4f}")
print(f"Oversampling Only:              {oversampling_results['f1_binary'] - baseline_results['f1_binary']:+.4f}")
print(f"Semi-Supervised Only:           {semi_only_results['f1_binary'] - baseline_results['f1_binary']:+.4f}")

# 이 부분은 plot 출력용이므로 주석 처리 하였습니다.
# print("\n" + "=" * 60)
# print("Oversampling Method Ablation Study")
# print("=" * 60)
#
# oversampling_ablation = {}
#
# fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# axes = axes.ravel()
#
# models_for_viz = [
#     ('Semi-Supervised + Oversampling', combined_results),
#     ('Oversampling Only', oversampling_results),
#     ('Semi-Supervised Only', semi_only_results),
#     ('Baseline', baseline_results)
# ]
#
# for i, (name, results) in enumerate(models_for_viz):
#     cm = confusion_matrix(val_test_y, results['predictions'])
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
#     axes[i].set_title(f'{name}')
#     axes[i].set_xlabel('Predicted')
#     axes[i].set_ylabel('Actual')
#
# plt.tight_layout()
# plt.show()

# Test 데이터 전처리
test_x = test_df.drop(columns=['ID'])
test_x_scaled = scaler.transform(test_x)

# 최고 성능 모델로 예측 (F1 Binary 기준)
best_model_name = max(models.keys(), key=lambda x: models[x]['f1_binary'])
best_model = {
    'Semi-Supervised + Oversampling': model_semi,
    'Oversampling Only': model_oversampling_only,
    'Semi-Supervised Only': model_semi_only,
    'Baseline': model_baseline
}[best_model_name]

print(f"\n최고 성능 모델: {best_model_name}")
print(f"F1 Binary Score: {models[best_model_name]['f1_binary']:.4f}")

# 최종 예측
test_pred = best_model.predict(test_x_scaled)
test_pred = get_pred_label(test_pred)

print(f"Test predictions - Normal: {sum(test_pred == 0)}, Fraud: {sum(test_pred == 1)}")
print(f"Test contamination ratio: {sum(test_pred == 1) / len(test_pred):.6f}")

"""## Submission"""

submit = pd.read_csv('./sample_submission.csv')
submit['Class'] = test_pred
submit.to_csv('./submit_semi_supervised_oversampling.csv', index=False)