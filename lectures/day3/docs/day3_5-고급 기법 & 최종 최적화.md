# 고급 기법 & 최종 최적화

ID: 3-5
일차: 3
순서: 5
상태: Active

> ***선택 과목**: 시간이 남는 경우 진행합니다. Day 3–4까지 완료하면 Day 3의 핵심 목표는 달성한 것입니다.*
> 

**목표**: Data Augmentation, Ensemble, TTA로 성능 극한까지 끌어올리기

## **1. 현재 상황 & 목표**

```
Day 3-1 (Simple CNN):    98.0%  → Baseline
Day 3-2 (ResNet):        99.1%  → Best Single Model
Day 3-3 (HPO):           98.8%  → Custom Tuned
Day 3-4 (Kaggle Submit): 98.9%  → 실전 제출

현재 순위: Top 10% 내외
목표:      Top 5% 도전 (99.3%+)
```

### **1.1 고급 기법 Overview**

```mermaid
graph TD
    A["현재 모델<br/>98.9%"] --> B["Data Augmentation<br/>+0.2~0.5%"]
    A --> C["Model Ensemble<br/>+0.3~0.7%"]
    A --> D["TTA<br/>+0.1~0.3%"]
    B --> E["최종 모델<br/>99.3%+ 목표"]
    C --> E
    D --> E

    style A fill:#e1f5fe
    style E fill:#bfb
```

## **2. Data Augmentation**

### **2.1 왜 필요한가?**

```mermaid
graph LR
    A["고정된 42,000개"] --> B["매 Epoch 동일 이미지"]
    B --> C["Overfitting 위험"]

    D["Augmentation 적용"] --> E["매 Epoch 변형된 이미지"]
    E --> F["일반화 능력 향상"]

    style C fill:#fbb
    style F fill:#bfb
```

**MNIST에 적합한 Augmentation**: 숫자는 상하 반전하면 의미가 바뀌므로(6↔9), 제한적으로 적용합니다.

```
✅ 적합: 소폭 회전(±10°), 이동(10%), 확대/축소(10%)
❌ 부적합: 좌우 반전(1↔  ), 상하 반전(6↔9), 과도한 회전
```

### **2.2 구현**

```python
datagen = ImageDataGenerator(
    rotation_range=10,       # ±10도 회전
    width_shift_range=0.1,   # 좌우 10% 이동
    height_shift_range=0.1,  # 상하 10% 이동
    zoom_range=0.1,          # 10% 확대/축소
    fill_mode='nearest'
)
```

![image.png](%EA%B3%A0%EA%B8%89%20%EA%B8%B0%EB%B2%95%20&%20%EC%B5%9C%EC%A2%85%20%EC%B5%9C%EC%A0%81%ED%99%94/image.png)

Augmentation은 `model.fit(datagen.flow(...))` 형태로 학습 시 실시간 적용됩니다. 별도 저장 없이 매 배치마다 새로운 변형을 만들어냅니다.

## **3. Model Ensemble**

### **3.1 왜 Ensemble이 효과적인가?**

```mermaid
graph TD
    A["모델 A: 이미지 X를 '3'으로 예측 (틀림)"]
    B["모델 B: 이미지 X를 '8'로 예측 (맞음)"]
    C["모델 C: 이미지 X를 '8'로 예측 (맞음)"]
    A --> D["다수결<br/>→ '8' ✅"]
    B --> D
    C --> D

    style D fill:#bfb
```

개별 모델의 오류 패턴이 서로 다를 때 Ensemble의 효과가 극대화됩니다. 서로 다른 아키텍처를 조합할수록 좋습니다.

### **3.2 MLflow에서 모델 로드**

```python
# 각 모델의 run_id를 MLflow에서 조회
model_names = ['Custom_Hybrid_Best', 'ResNet_style', 'VGG_style']
models = {}

for name in model_names:
    runs = mlflow.search_runs(
        filter_string=f"params.model = '{name}'",
        order_by=["metrics.final_val_accuracy DESC"]
    )
    run_id = runs.iloc[0]['run_id']
    models[name] = mlflow.keras.load_model(f"runs:/{run_id}/model")
    print(f"✅ {name} 로드 완료")
```

> *⚠️ Custom Hybrid 모델은 `SelfAttention` 클래스 정의 후 로드해야 합니다.*
> 

### **3.3 Simple Average Ensemble**

```python
# 3개 모델의 예측 확률을 평균
preds = [model.predict(X_val, batch_size=256) for model in models.values()]
ensemble_pred = np.mean(preds, axis=0)
ensemble_labels = np.argmax(ensemble_pred, axis=1)

ensemble_acc = np.mean(ensemble_labels == y_val)
print(f"Ensemble Val Accuracy: {ensemble_acc:.4f}")
```

### **3.4 Weighted Ensemble (선택)**

모델 성능에 비례해 가중치를 줄 수 있습니다.

```python
weights = {'Custom_Hybrid_Best': 0.4, 'ResNet_style': 0.4, 'VGG_style': 0.2}

weighted_pred = sum(
    w * models[name].predict(X_val, batch_size=256)
    for name, w in weights.items()
)
```

![image.png](%EA%B3%A0%EA%B8%89%20%EA%B8%B0%EB%B2%95%20&%20%EC%B5%9C%EC%A2%85%20%EC%B5%9C%EC%A0%81%ED%99%94/image%201.png)

## **4. Test-Time Augmentation (TTA)**

### **4.1 TTA란?**

```mermaid
graph LR
    A["원본 이미지"] --> B["TTA 없이<br/>1회 예측"]
    B --> F["단일 예측값"]

    A --> C["약간 회전 →<br/>예측1"]
    A --> D["약간 이동 →<br/>예측2"]
    A --> E["약간 확대 →<br/>예측3"]
    C --> G["평균 → 안정적 예측값"]
    D --> G
    E --> G

    style F fill:#fdd
    style G fill:#bfb
```

학습 시 Augmentation을 적용한 것처럼, **추론 시에도 여러 변형 이미지의 예측을 평균**냅니다. 모델의 불확실성을 줄이고 강건한 예측이 가능합니다.

### **4.2 구현**

```python
def predict_with_tta(model, images, n_augment=5):
    tta_datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.1
    )
    predictions = [model.predict(images, batch_size=256, verbose=0)]
    for _ in range(n_augment):
        aug_iter = tta_datagen.flow(images, batch_size=len(images), shuffle=False)
        aug_imgs = next(aug_iter)
        predictions.append(model.predict(aug_imgs, batch_size=256, verbose=0))
    return np.mean(predictions, axis=0)
```

![image.png](%EA%B3%A0%EA%B8%89%20%EA%B8%B0%EB%B2%95%20&%20%EC%B5%9C%EC%A2%85%20%EC%B5%9C%EC%A0%81%ED%99%94/image%202.png)

**TTA 횟수 선택 기준**: 510회가 성능/속도 균형에 최적입니다. 15회 이상은 개선이 미미합니다.

## **5. 최종 제출**

Weighted Ensemble + TTA를 결합한 최강 전략으로 최종 제출합니다.

```python
final_predictions = []
for name, model in models.items():
    pred = predict_with_tta(model, X_test, n_augment=10)
    final_predictions.append(pred * weights[name])

final_pred = np.sum(final_predictions, axis=0)
final_labels = np.argmax(final_pred, axis=1)

final_submission = pd.DataFrame({
    'ImageId': range(1, len(final_labels) + 1),
    'Label': final_labels
})
final_submission.to_csv('final_submission_day3_5.csv', index=False)
```

## **6. Day 3 전체 성능 분석**

![image.png](%EA%B3%A0%EA%B8%89%20%EA%B8%B0%EB%B2%95%20&%20%EC%B5%9C%EC%A2%85%20%EC%B5%9C%EC%A0%81%ED%99%94/image%203.png)

```css
Day 3-1 Simple CNN:          98.0%   +0.0%
Day 3-2 ResNet-style:        99.1%   +1.1%
Day 3-3 Custom HPO:          98.8%   +0.8%
Day 3-4 Kaggle Score:        98.9%   실전 확인
Day 3-5 Ensemble + TTA:      99.3%+  +0.4~0.5%
```

**각 기법의 기여:**

| **기법** | **효과** | **핵심 원리** |
| --- | --- | --- |
| Data Augmentation | +0.20.5% | 데이터 다양성 확보 → 일반화 |
| Ensemble | +0.30.7% | 오류 패턴 상호 보정 |
| TTA | +0.10.3% | 추론 불확실성 감소 |

## **7. Day 3 핵심 교훈**

```mermaid
graph TD
    A["Day 3 전체 여정"] --> B["아키텍처가 전부가 아니다<br/>Custom Hybrid: 94% → 98.8%<br/>튜닝으로 극적 개선"]
    A --> C["체계적 실험 관리<br/>MLflow 없었다면<br/>어떤 설정이 좋았는지 기억 불가"]
    A --> D["실전 경험의 가치<br/>Kaggle 제출 → Public Score<br/>실제 성능 객관적 확인"]
    A --> E["고급 기법의 한계<br/>Ensemble+TTA도 결국<br/>좋은 모델이 기반"]

    style A fill:#e1f5fe
```

**실무 적용 로드맵:**

1. Baseline 빠르게 구축

2. MLflow로 모든 실험 추적

3. 아키텍처 탐색

4. 하이퍼파라미터 최적화 (Optuna)

5. Augmentation + Ensemble + TTA로 마무리

## **✅ 체크리스트**

- [ ]  Data Augmentation 시각화 확인 (원본 vs 증강)
- [ ]  Augmentation 적용하여 재학습 완료
- [ ]  MLflow에서 3개 모델 로드 완료
- [ ]  Simple Average Ensemble 구현 및 성능 확인
- [ ]  TTA 함수 구현 (n_augment=5)
- [ ]  TTA 횟수별 성능 실험 (1, 3, 5, 10, 15)
- [ ]  Weighted Ensemble + TTA 최종 예측 완료
- [ ]  final_submission_day3_5.csv 생성 및 Kaggle 재제출
- [ ]  Day 3 전체 성능 향상 그래프 확인

🎉 **Day 3 완료! 수고하셨습니다.**