# MobileNetV2 Transfer Learning

ID: 5-2
일차: 5
순서: 2
상태: Active

**목표**: MobileNetV2 Transfer Learning으로 경량 제스처 인식 모델 97%+ 달성

## **1. Depthwise Separable Convolution**

MobileNetV2의 핵심: 연산을 두 단계로 분리해 계산량을 ~7배 줄입니다.

```mermaid
graph TB
    subgraph Standard["Standard Convolution"]
        A1["Input (H×W×C_in)"] --> B1["3×3 Conv<br/>C_in × C_out filters<br/>모든 채널 동시 처리"]
        B1 --> C1["Output (H×W×C_out)"]
        style B1 fill:#fbb
    end
    subgraph DSConv["Depthwise Separable Convolution"]
        A2["Input (H×W×C_in)"] --> B2["① 3×3 Depthwise Conv<br/>채널별 독립 처리<br/>C_in filters"]
        B2 --> B3["② 1×1 Pointwise Conv<br/>채널 믹싱<br/>C_in → C_out"]
        B3 --> C2["Output (H×W×C_out)"]
        style B2 fill:#99ccff
        style B3 fill:#bfb
    end
```

**연산량 비교** (224×224, 3→32채널, 3×3 kernel):

|  | **Standard Conv** | **Depthwise Separable** |
| --- | --- | --- |
| **연산량** | 43,614,208 | 6,171,648 |
| **파라미터** | 864 | 123 |
| **비율** | 1× | **약 1/7** |

## **2. MobileNetV2 아키텍처**

### **2.1 Inverted Residual Block**

```mermaid
graph LR
    subgraph ResNet["일반 ResNet Block"]
        A1["넓음 256"] --> B1["좁음 64<br/>(Bottleneck)"]
        B1 --> C1["넓음 256"]
        A1 -. "Skip" .-> C1
    end
    subgraph MobileNet["MobileNetV2 Inverted Residual"]
        A2["좁음 24"] --> B2["넓음 144<br/>(Expansion ×6)"]
        B2 --> C2["좁음 24"]
        A2 -. "Skip" .-> C2
        style B2 fill:#99ccff
    end
```

- **ResNet**: Wide → Narrow → Wide (중간이 좁은 Bottleneck)
- **MobileNetV2**: Narrow → Wide → Narrow (Expansion Factor=6으로 중간을 넓혀 표현력 확보)
- 마지막 1×1 Conv에 **ReLU 없음** (Linear Bottleneck) — 저차원에서 정보 손실 방지

### **2.2 전체 구조**

```mermaid
graph TD
    A["Input 224×224×3"] --> B["Conv2D stride=2<br/>112×112×32"]
    B --> C["Bottleneck ×1<br/>t=1, c=16"]
    C --> D["Bottleneck ×2+3<br/>t=6, c=24, 32, stride=2"]
    D --> E["Bottleneck ×4+3+3<br/>t=6, c=64 / 96 / 160, stride=2"]
    E --> F["Bottleneck ×1<br/>t=6, c=320"]
    F --> G["Conv2D 1×1<br/>7×7×1280"]
    G --> H["GlobalAveragePooling<br/>1280"]
    H --> I["Custom Classifier<br/>19 classes"]
    style A fill:#e1f5fe
    style I fill:#ffccbc
```

총 파라미터 **3.5M** · 크기 **~14MB** (FP32) · 추론 **~1.8ms**

## **3. Data Augmentation**

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),        # 좌우 반전 ✅ (손 방향 다양화)
    layers.RandomRotation(0.1),             # ±10도 회전 ✅
    layers.RandomZoom(0.1),                 # ±10% 확대/축소 ✅
    layers.RandomTranslation(0.1, 0.1),     # ±10% 이동 ✅
])
```

![image.png](MobileNetV2%20Transfer%20Learning/image.png)

**구현 포인트**: `shuffle → batch → augment` 순서입니다. 배치를 먼저 만든 뒤 augment_batch를 map합니다.

## **4. MobileNetV2용 전처리**

Baseline CNN(01 정규화)과 달리 MobileNetV2는 **ImageNet 기준 전처리**가 필요합니다.

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_and_preprocess_mobilenet(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)   # [0,255] → [-1, 1]
    return img, label
```

`preprocess_input`은 `[0,255] → [-1, 1]`로 변환합니다. Pretrained 가중치가 이 범위를 가정하고 학습됐으므로 반드시 적용해야 합니다.

## **5. 모델 구축**

```python
base_model = MobileNetV2(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
base_model.trainable = False   # Phase 1: 완전 동결

inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)   # training=False → BN이 ImageNet 통계 사용
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(gesture_names), activation='softmax')(x)  # 19 classes
model = Model(inputs, outputs)
```

## **6. 2단계 Transfer Learning**

```mermaid
graph LR
    A["Phase 1<br/>Feature Extraction<br/>base_model.trainable=False<br/>lr=1e-3 · 5 epochs"] --> B["Phase 2<br/>Fine-tuning<br/>base_model.trainable=True<br/>lr=1e-5 · 10 epochs"]
    B --> C["최종 모델<br/>97.6% ✅"]
    style A fill:#fff9c4
    style B fill:#ffe082
    style C fill:#bfb
```

**Phase 1 — Feature Extraction:**

```python
model.compile(optimizer=Adam(learning_rate=1e-3), ...)
with mlflow.start_run(run_name='MobileNetV2_Phase1_FeatureExtraction'):
    mlflow.log_params({'phase': 'feature_extraction', 'base_trainable': False,
                       'learning_rate': 1e-3, 'epochs': 5})
    history_phase1 = model.fit(train_dataset, epochs=5, ...)
```

**Phase 2 — Fine-tuning:**

```python
base_model.trainable = True   # 전체 해제
model.compile(optimizer=Adam(learning_rate=1e-5), ...)  # LR 100배 감소!
with mlflow.start_run(run_name='MobileNetV2_Phase2_FineTuning'):
    mlflow.log_params({'phase': 'fine_tuning', 'base_trainable': True,
                       'learning_rate': 1e-5, 'epochs': 10})
    history_phase2 = model.fit(train_dataset, epochs=10, ...)
```

lr=1e–5(100배 감소)를 쓰는 이유: 큰 LR로 업데이트하면 ImageNet에서 학습한 특징을 망가뜨립니다.

## **7. 학습 결과 & 비교**

![image.png](MobileNetV2%20Transfer%20Learning/image%201.png)

![image.png](MobileNetV2%20Transfer%20Learning/image%202.png)

```
================================================================================
  Baseline CNN vs MobileNetV2
================================================================================
       Model Parameters  Size (MB)  Val Accuracy (%) Test Accuracy (%)  Latency (ms)        FPS
Baseline CNN         2M   8.000000         73.520000                 -     15.000000  67.000000
 MobileNetV2       3.5M  27.990143         97.580224         97.519622      1.798715 555.952422
================================================================================

개선:
  Accuracy: +24.06%p
  Model Size: +19.99MB
  Latency: +-13.20ms
```

```
                Baseline CNN    MobileNetV2
Val Accuracy:   73.52%          97.58%  (+24.06%p)
Model Size:     ~8 MB           ~28 MB
Latency:        ~15 ms          ~1.8 ms  (-88%)
FPS:            ~67             ~556
```

정확도는 크게 향상되고 추론 속도도 8배 이상 빨라졌습니다. 크기는 커졌지만 Day 5–3 Quantization으로 7MB까지 줄입니다.

## **✅ 체크리스트**

- [ ]  Depthwise Separable Conv 원리 이해 (~1/7 연산량)
- [ ]  Inverted Residual Block 개념 이해 (Narrow→Wide→Narrow)
- [ ]  Data Augmentation 4가지 적용 및 시각화
- [ ]  MobileNetV2용 `preprocess_input` 적용 ([–1,1] 범위)
- [ ]  Phase 1: Feature Extraction (lr=1e–3, 5 epochs, base frozen)
- [ ]  Phase 2: Fine-tuning (lr=1e–5, 10 epochs, base unfrozen)
- [ ]  Phase 1+2 결합 학습 곡선 시각화
- [ ]  Val Accuracy 97%+ 달성
- [ ]  Latency ~1.8ms, FPS ~556 확인
- [ ]  MLflow에 Phase 1, Phase 2 각각 기록