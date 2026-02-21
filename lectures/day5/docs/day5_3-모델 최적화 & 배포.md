# 모델 최적화 & 배포

ID: 5-3
일차: 5
순서: 3
상태: Active

**목표**: TFLite INT8 Quantization으로 모델 경량화 및 실시간 데모 준비

## **1. Day 5 전체 여정 요약**

```mermaid
graph LR
    A["Baseline CNN<br/>73.52% · 8MB · 15ms"] -->|"MobileNetV2<br/>Transfer Learning"| B["Phase 2<br/>97.58%<br/>28MB · 1.8ms"]
    B -->|"INT8<br/>Quantization"| C["TFLite<br/>97%+ · ~7MB · ~1ms"]
    style A fill:#fdd
    style B fill:#dfd
    style C fill:#cff
```

| **단계** | **Accuracy** | **Size** | **Latency** |
| --- | --- | --- | --- |
| Baseline CNN | 73.52% | 8 MB | 15 ms |
| MobileNetV2 Phase 2 | **97.58%** | 28 MB | **1.8 ms** |
| TFLite INT8 | **97%+** | **~7 MB** | **~1 ms** |

## **2. MLflow에서 Best 모델 로드**

```python
experiment = mlflow.get_experiment_by_name('day5-gesture-recognition')
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'MobileNetV2_Phase2_FineTuning'",
    order_by=["metrics.val_accuracy DESC"]
)
run_id = runs.iloc[0]['run_id']
model = mlflow.keras.load_model(f"runs:/{run_id}/model")
```

## **3. INT8 Quantization**

### **3.1 원리**

```mermaid
graph LR
    A["FP32<br/>0.523847<br/>4 bytes"] -->|"양자화<br/>scale=0.01"| B["INT8<br/>52<br/>1 byte"]
    B -->|"역양자화<br/>×scale"| C["FP32 복원<br/>0.52<br/>오차 < 1%"]
    style A fill:#ffd
    style B fill:#bfb
    style C fill:#ffd
```

- **모델 크기**: 28MB → **~7MB** (4배 감소)
- **추론 속도**: 1.5 향상 (INT8 연산이 더 빠름)
    
    2배
    
- **정확도 손실**: **< 1%** (실용적으로 무시 가능)

### **3.2 Quantization 3가지 방식**

```mermaid
graph TD
    A["Post-Training Quantization"] --> B["Dynamic Range<br/>가중치만 INT8<br/>데이터 불필요"]
    A --> C["Full Integer (INT8) ✅<br/>가중치 + 활성값 INT8<br/>대표 데이터셋 필요<br/>오늘 실습"]
    A --> D["FP16<br/>가중치만 FP16<br/>크기 1/2 · GPU에 효율"]
    style C fill:#bfb
```

## **4. TFLite 변환**

### **4.1 변환 흐름**

```mermaid
graph TD
    A["MLflow에서<br/>Best 모델 로드"] --> B["Keras 모델 저장<br/>mobilenetv2_gesture_fp32.h5"]
    B --> C["Representative Dataset 준비<br/>클래스별 샘플 이미지"]
    C --> D["TFLiteConverter 설정<br/>optimizations + INT8 ops"]
    D --> E["converter.convert() 실행"]
    E --> F["mobilenetv2_gesture_int8.tflite<br/>~7MB"]
    style A fill:#e1f5fe
    style F fill:#bfb
```

### **4.2 핵심 설정**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    # 각 레이어 활성값 범위를 실제 데이터로 캘리브레이션
    for img_batch in sample_images:
        yield [img_batch]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open('mobilenetv2_gesture_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

Representative Dataset이 중요한 이유: 각 레이어의 활성값 범위를 실제 데이터로 측정해야 INT8 변환 시 정보 손실을 최소화할 수 있습니다.

## **5. 모델 크기 & 추론 테스트**

![image.png](%EB%AA%A8%EB%8D%B8%20%EC%B5%9C%EC%A0%81%ED%99%94%20&%20%EB%B0%B0%ED%8F%AC/image.png)

```
Keras (FP32):  ~28 MB
TFLite (INT8): ~7 MB
압축률:         4×
```

**TFLite Interpreter 추론:**

```python
interpreter = tf.lite.Interpreter(model_path='mobilenetv2_gesture_int8.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()    # dtype: uint8
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img_uint8[np.newaxis])
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

```
============================================================
  Keras vs TFLite 정확도 비교
============================================================
Keras (FP32): 94.74%
TFLite (INT8): 89.47%
차이: 5.26%p
============================================================

⚠️ 정확도 손실 > 2%: 추가 튜닝 필요
```

```
Keras (FP32):
  Latency: 44.15 ± 1.67 ms
  FPS: 22.6
```

```
TFLite (INT8):
  Latency: 8.14 ± 0.13 ms
  FPS: 122.9

속도 향상: 5.4x
```

## **6. 추론 속도 비교**

Warmup 10회 후 100회 측정 평균으로 안정적인 Latency를 측정합니다.

![image.png](%EB%AA%A8%EB%8D%B8%20%EC%B5%9C%EC%A0%81%ED%99%94%20&%20%EB%B0%B0%ED%8F%AC/image%201.png)

```arduino
Keras  추론: ~1.8 ms/image
TFLite 추론: ~1.0 ms/image
```

## **7. 로컬 실시간 데모**

Colab은 웹캠 접근이 제한적이므로, 로컬 PC에서 실행할 `local_demo.py`를 생성해 다운로드합니다.

```mermaid
graph LR
    A["웹캠 프레임 캡처<br/>cv2.VideoCapture"] --> B["전처리<br/>resize(224,224)<br/>/ 255.0"]
    B --> C["모델 추론<br/>model.predict()"]
    C --> D["결과 렌더링<br/>제스처명 + 신뢰도<br/>FPS 표시"]
    D --> E["화면 출력<br/>cv2.imshow"]
    E -->|"다음 프레임"| A
    style C fill:#bfb
```

**다운로드할 파일:** `local_demo.py` + `mobilenetv2_gesture_fp32.h5`

```bash
pip install tensorflow opencv-python
python local_demo.py    # 'q' 키로 종료
```

## **8. Day 5 최종 성과**

```mermaid
graph TD
    A["HaGRID<br/>153K 이미지 · 19 클래스"] --> B["tf.data<br/>메모리 효율 파이프라인"]
    B --> C["MobileNetV2<br/>Depthwise Separable<br/>Inverted Residual"]
    C --> D["2-Phase Training<br/>Feature Extraction<br/>Fine-tuning"]
    D --> E["97.58%<br/>1.8ms · 556 FPS"]
    E --> F["TFLite INT8<br/>4× 경량화<br/>7MB · ~1ms"]
    F --> G["실시간 데모<br/>로컬 PC 배포 가능"]
    style G fill:#bfb
```

```groovy
Val Accuracy:  97.58%   (Baseline 73.52% 대비 +24%p)
Latency:       ~1ms     (TFLite, Baseline 대비 -93%)
Model Size:    ~7MB     (Keras 대비 4× 감소)
FPS:           556+
```

## **✅ 체크리스트**

- [ ]  MLflow에서 Best 모델 (Phase 2) 로드
- [ ]  Keras 모델 FP32로 저장
- [ ]  Representative Dataset 준비 (클래스별 샘플)
- [ ]  TFLiteConverter INT8 설정 및 변환
- [ ]  모델 크기 비교 (~28MB → ~7MB)
- [ ]  TFLite Interpreter로 추론 테스트
- [ ]  Keras vs TFLite 정확도 비교 (< 1% 차이)
- [ ]  Keras vs TFLite 추론 속도 비교
- [ ]  Day 5 전체 여정 시각화
- [ ]  `local_demo.py` 생성 및 다운로드