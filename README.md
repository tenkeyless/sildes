# Slidev 슬라이드 (충남대 딥러닝 프로젝트 강의)

[Docker](https://www.docker.com/) 기반 Slidev 슬라이드 실행 환경입니다.

## 요구 사항

- **Docker만** 있으면 됩니다 (Node/npm 설치 불필요, Apple Silicon M1 호환)

## 슬라이드 실행

### 방법 1: 웹 런처 — Docker로 실행 (권장, npm 불필요)

**메인 폴더(slides/)에서** 아래만 실행하면 됩니다. 이미지가 없으면 스크립트가 자동으로 빌드합니다.

```bash
chmod +x run-launcher.sh
./run-launcher.sh
```

브라우저에서 **http://localhost:3040** 을 열고, 원하는 슬라이드 파일을 클릭하면 Slidev가 실행되며 **http://localhost:3030** 이 새 탭에서 열립니다.

### 방법 1-2: 웹 런처 — npm으로 실행

Node가 설치되어 있다면 메인 폴더에서:

```bash
npm run launcher
```

### 방법 2: 스크립트 사용 (선택)

웹 런처 대신 터미널에서 파일명만 넘겨서 띄우고 싶을 때 사용합니다. 한 번만 실행 권한을 부여합니다.

```bash
chmod +x run-slidev.sh
./run-slidev.sh Day1-3_BERT_Slides.md
```

### 방법 3: Docker 이미지 수동 빌드

이미지를 직접 빌드할 때:

```bash
docker build -t my-slidev:m1 .
docker build -t slidev-launcher slidev-launcher/
```

### 방법 4: docker run 직접 사용

```bash
docker run --name slidev-m1 --rm -it \
    -v "$(pwd):/slidev" \
    -p 3030:3030 \
    my-slidev:m1 slidev <슬라이드파일.md> --remote
```

실행 후 브라우저에서 **http://localhost:3030** 으로 접속합니다.

## 슬라이드 파일

| 파일명 | 설명 |
|--------|------|
| `Day1-2_TF-IDF_Slides.md` | TF-IDF 슬라이드 |
| `Day1-3_BERT_Slides.md` | BERT 슬라이드 |
| `Day2-1_Seq2Seq_Slides.md` | Seq2Seq 슬라이드 |
| `day1_1-slide-환경 설정 & MLflow 세팅.md` | Day 1-1 슬라이드 |
| `day1_2-slide-문제 이해와 베이스라인.md` | Day 1-2 슬라이드 |
| `day1_3-slide-Transformer 모델.md` | Day 1-3 슬라이드 |
| `day2_1-slide-Seq2Seq 기초 — 데이터 탐색 & BLEU Score.md` | Day 2-1 슬라이드 |
| `day2_2-slide-LSTM 기반 Seq2Seq 모델.md` | Day 2-2 슬라이드 |
| `day2_3-slide-Transformer와 MarianMT.md` | Day 2-3 슬라이드 |
| `day3_1-slide-MNIST 데이터 탐색 & 베이스라인 모델.md` | Day 3-1 슬라이드 |
| `day3_2-slide-다양한 CNN 아키텍처 비교.md` | Day 3-2 슬라이드 |
| `day3_3-slide-하이퍼파라미터 튜닝 with Optuna & MLflow.md` | Day 3-3 슬라이드 |
| `day3_4-slide-Kaggle 제출 & 결과 분석.md` | Day 3-4 슬라이드 |
| `day3_5-slide-고급 기법 & 최종 최적화.md` | Day 3-5 슬라이드 (선택) |
| `day4_1-slide-COVID-19 흉부 X-ray EDA.md` | Day 4-1 슬라이드 |
| `day4_2-slide-Transfer Learning with ResNet50.md` | Day 4-2 슬라이드 |
| `day4_3-slide-Class Imbalance 처리.md` | Day 4-3 슬라이드 |
| `day4_4-slide-Grad-CAM & 최종 평가.md` | Day 4-4 슬라이드 |
| `day5_1-slide-HaGRID 손 제스처 데이터 EDA.md` | Day 5-1 슬라이드 |

---

## 슬라이드 ↔ 노트북 흐름

각 강의는 **슬라이드 → 노트북 실습**을 번갈아 진행합니다.  
`🔥` 표시는 학생이 직접 채워야 하는 부분입니다.

---

### Day 1-1: 환경 설정 & MLflow 세팅

> 슬라이드: `day1_1-slide-환경 설정 & MLflow 세팅.md`  
> 노트북: `lectures/day1/start/day1_1-환경설정-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 1️⃣ Google Colab 환경 확인 (GPU) |
| ② | Dagshub란? / 왜 실험 기록? / MLflow + Dagshub 연동 구조 / 계정 생성 | — |
| ③ | — | 2️⃣ Dagshub & MLflow 설정 (🔥 repo_owner, repo_name) |
| ④ | MLflow 기본 로깅 구조 | — |
| ⑤ | — | 3️⃣ MLflow 첫 실험 로깅 테스트 (🔥 run_name, log_param, log_metric) |
| ⑥ | Dagshub UI 결과 확인 / Dacon 데이터 소개 | — |
| ⑦ | — | 4️⃣ Dacon 데이터 다운로드 |
| ⑧ | 오늘 배운 것 / 체크리스트 | — |

---

### Day 1-2: 문제 이해와 베이스라인

> 슬라이드: `day1_2-slide-문제 이해와 베이스라인.md`  
> 노트북: `lectures/day1/start/day1_2-베이스라인-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 0. 환경 재설정 |
| ② | 문제 정의 / 7개 토픽 / EDA에서 확인할 것들 | — |
| ③ | — | 1. 데이터 로드 |
| ④ | — | 2. 탐색적 데이터 분석 (EDA) |
| ⑤ | TF-IDF (BoW → TF-IDF → IDF 직관 → Char n-gram) | — |
| ⑥ | Logistic Regression (원리 → C 파라미터) | — |
| ⑦ | Accuracy의 함정 + Macro F1 / 데이터 누수 주의 | — |
| ⑧ | — | 3. 베이스라인 모델 (🔥 train_test_split, tfidf, model, 평가) |
| ⑨ | MLflow 실험 기록 (run_name 규칙) | — |
| ⑩ | — | 4. MLflow 실험 로깅 (🔥 run_name) |
| ⑪ | 실험 설계 원칙 / 목록 / 예상 패턴 | — |
| ⑫ | — | 5. 성능 개선 실험 |
| ⑬ | — | 6. 실험 결과 비교 |
| ⑭ | 베이스라인의 의미 / TF-IDF vs BERT / 체크리스트 | — |

---

### Day 1-3: Transformer 모델 (BERT)

> 슬라이드: `day1_3-slide-Transformer 모델.md`  
> 노트북: `lectures/day1/start/day1_3-BERT-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 0. 환경 재설정 → 1. 데이터 준비 |
| ② | TF-IDF 한계 / Transformer 아키텍처 / BERT 사전학습·Fine-tuning / 모델 선택 / Tokenization | — |
| ③ | — | 2. BERT 모델 및 Tokenizer 로드 (🔥 model_name, tokenizer, model) |
| ④ | Fine-tuning 개념 / Learning Rate / Epochs / Warmup Steps | — |
| ⑤ | — | 3. BERT Fine-tuning (🔥 TrainingArguments 6개 항목, compute_metrics, Trainer) |
| ⑥ | — | 4. 모델 평가 (🔥 eval_result, predictions, y_pred, y_true) |
| ⑦ | MLflow 실험 기록 | — |
| ⑧ | — | 5. MLflow 실험 로깅 (🔥 run_name) |
| ⑨ | TF-IDF vs BERT / 오늘 배운 것 / 체크리스트 | — |

---

### Day 2-1: Seq2Seq 기초 — 데이터 탐색 & BLEU Score

> 슬라이드: `day2_1-slide-Seq2Seq 기초 — 데이터 탐색 & BLEU Score.md`  
> 노트북: `lectures/day2/start/day2_1-seq2seq-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 0. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | 기계번역 역사 / Encoder-Decoder 구조 / Teacher Forcing / Attention | — |
| ③ | — | 1. 데이터 로드 & EDA (문장 길이 분포, Zipf's Law) |
| ④ | 토큰화 & Vocabulary / 특수 토큰 / 데이터 누수 주의 | — |
| ⑤ | — | Vocabulary 구축 (🔥 build(), encode()) |
| ⑥ | BLEU Score / n-gram / 계산 예시 / 한계 | — |
| ⑦ | — | 5. BLEU Score |
| ⑧ | 패딩이 필요한 이유 | — |
| ⑨ | — | 6. Seq2Seq 입력 데이터 준비 (🔥 encode_sentence truncate/pad) |
| ⑩ | MLflow 기록 내용 | — |
| ⑪ | — | 7. MLflow 실험 기록 (🔥 run_name) |
| ⑫ | 오늘 배운 것 / 체크리스트 | — |

---

### Day 2-2: LSTM 기반 Seq2Seq 모델

> 슬라이드: `day2_2-slide-LSTM 기반 Seq2Seq 모델.md`  
> 노트북: `lectures/day2/start/day2_2-lstm_seq2seq-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 1. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | LSTM (왜 필요? / 4개 게이트) / Seq2Seq 전체 흐름 / Teacher Forcing / 구조도 | — |
| ③ | — | 2. Day 2-1 데이터 로드 |
| ④ | — | 3. PyTorch Dataset & DataLoader |
| ⑤ | Loss & Gradient Clipping / 학습 루프 흐름 (sequenceDiagram) | — |
| ⑥ | — | 4. LSTM Seq2Seq 모델 구현 (🔥 Encoder.forward, Decoder.forward, Seq2Seq decoder 호출) |
| ⑦ | — | 5. 학습 준비 & 6. 학습 루프 |
| ⑧ | — | 7. 실험 1: Baseline (🔥 run_name) |
| ⑨ | — | 8. 학습 곡선 시각화 |
| ⑩ | Attention 필요성 / Bahdanau Attention / Decoder 구조도 | — |
| ⑪ | — | 9. Attention 구현 & 실험 2 (선택) |
| ⑫ | Greedy Decoding / 오늘 배운 것 / 체크리스트 | — |

---

### Day 2-3: Transformer와 MarianMT

> 슬라이드: `day2_3-slide-Transformer와 MarianMT.md`  
> 노트북: `lectures/day2/start/day2_3-transformer_marianmt-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 1. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | RNN 한계 / Self-Attention / Multi-Head Attention / Positional Encoding | — |
| ③ | Transformer 전체 구조 / Encoder Layer / Decoder Layer / Encoder→모든 Decoder | — |
| ④ | MarianMT란? / 왜 쓰나? | — |
| ⑤ | — | 2. 데이터 준비 |
| ⑥ | — | 3. MarianMT 모델 로드 (🔥 model_name) |
| ⑦ | Greedy vs Beam Search / Beam Search 예시 | — |
| ⑧ | — | 4. Zero-Shot 번역 (🔥 model.generate, batch_decode) |
| ⑨ | — | 5. Beam Search 실험 |
| ⑩ | — | 6. BLEU Score 평가 |
| ⑪ | MLflow 기록 내용 | — |
| ⑫ | — | 7. MLflow 실험 기록 (🔥 run_name) |
| ⑬ | — | 8. Day 2 전체 모델 성능 비교 |
| ⑭ | 핵심 교훈 / 오늘 배운 것 / 체크리스트 | — |

---

### Day 3-1: MNIST 데이터 탐색 & 베이스라인 모델

> 슬라이드: `day3_1-slide-MNIST 데이터 탐색 & 베이스라인 모델.md`  
> 노트북: `lectures/day3/start/day3_1-mnist_baseline-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 1. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | Kaggle이란? / Digit Recognizer 흐름 / Public vs Private | — |
| ③ | MNIST 데이터 개요 | — |
| ④ | — | 2. 데이터 로드 & 3. EDA |
| ⑤ | 클래스 분포 / 샘플 이미지 / 픽셀 분포 / 혼동 숫자 쌍 (이미지) | — |
| ⑥ | 정규화 & Reshape & Train/Val Split 개념 | — |
| ⑦ | — | 4. 데이터 전처리 (🔥 normalize, reshape, train_test_split) |
| ⑧ | CNN 아키텍처 (mermaid) / Conv2D·MaxPooling·Dropout 설명 | — |
| ⑨ | — | 5. Simple CNN 구현 (🔥 Sequential 레이어 채우기) |
| ⑩ | — | 6. 모델 학습 (🔥 run_name) |
| ⑪ | — | 7. 결과 분석 |
| ⑫ | 학습 결과 (이미지) / 오늘 배운 것 / 체크리스트 | — |

---

### Day 3-2: 다양한 CNN 아키텍처 비교

> 슬라이드: `day3_2-slide-다양한 CNN 아키텍처 비교.md`  
> 노트북: `lectures/day3/start/day3_2-cnn_architectures-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 1. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | Simple CNN 한계 / 아키텍처 로드맵 / 핵심 기법 표 | — |
| ③ | — | 2. 데이터 로드 |
| ④ | LeNet-5 역사 / 구조 | — |
| ⑤ | — | 3. LeNet-5 구현 (🔥 Sequential 레이어) + run_name |
| ⑥ | VGG 철학 / 구조 / Batch Normalization | — |
| ⑦ | — | 4. VGG-style 구현 (🔥 BatchNormalization 위치) + run_name |
| ⑧ | Skip Connection 개념 / Residual Block | — |
| ⑨ | — | 5. ResNet-style 구현 (🔥 Add([fx, x])) + run_name |
| ⑩ | Squeeze-and-Excitation Block | — |
| ⑪ | — | 6. SE-CNN 구현 (🔥 Excitation + Scale) + run_name |
| ⑫ | Custom Hybrid 철학 / gamma 역할 | — |
| ⑬ | — | 7. Custom Hybrid 구현 (🔥 gamma * out + x) + run_name |
| ⑭ | — | 8. 모델 성능 비교 |
| ⑮ | 실험 결과 (이미지) / 예상 밖 결과 분석 / 오늘 배운 것 / 체크리스트 | — |

---

### Day 3-3: 하이퍼파라미터 튜닝 with Optuna & MLflow

> 슬라이드: `day3_3-slide-하이퍼파라미터 튜닝 with Optuna & MLflow.md`  
> 노트북: `lectures/day3/start/day3_3-hyperparameter_tuning-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 1. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | Day 3-2 교훈 / 실패 원인 / 튜닝 전략 3단계 | — |
| ③ | — | 2. 데이터 로드 & 3. Custom Hybrid 모델 |
| ④ | LR 탐색 개념 + 예상 결과 / LR Grid Search 결과 (이미지) | — |
| ⑤ | Gamma 초기화 탐색 / Gamma Grid Search 결과 | — |
| ⑥ | — | 4. Manual Grid Search (🔥 lr_candidates, gamma_candidates) |
| ⑦ | Optuna란? / 탐색 공간 표 / MLflow 통합 코드 | — |
| ⑧ | — | 5. Optuna 자동 탐색 (🔥 trial.suggest_* 6개 + n_trials) |
| ⑨ | Optimization History / Parameter Importance / Parallel Coordinate (이미지) | — |
| ⑩ | — | 6. 결과 시각화 & 7. Best Model 재학습 (🔥 run_name) |
| ⑪ | — | 8. Before vs After 비교 (이미지) |
| ⑫ | 핵심 교훈 / 오늘 배운 것 / 체크리스트 | — |

---

### Day 3-4: Kaggle 제출 & 결과 분석

> 슬라이드: `day3_4-slide-Kaggle 제출 & 결과 분석.md`  
> 노트북: `lectures/day3/start/day3_4-kaggle_submission-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 환경 설정 (🔥 repo_owner, repo_name) |
| ② | 제출 전체 흐름 (mermaid) / Best Model 로드 개념 | — |
| ③ | — | 1. Test Set 예측 & 2. submission.csv (🔥 ImageId·Label DataFrame 생성) |
| ④ | Web UI 제출 방법 (이미지) / API 토큰 발급 & Colab 보안 비밀 (이미지) | — |
| ⑤ | — | 3. Kaggle 제출 (🔥 API message) |
| ⑥ | 제출 결과 확인 (이미지 — Public Score 0.98864) | — |
| ⑦ | 왜 Error Analysis인가? | — |
| ⑧ | — | 4. Error Analysis (🔥 val_pred_labels = np.argmax) |
| ⑨ | 틀린 이미지 (이미지) / 클래스별 오류율 (이미지) | — |
| ⑩ | — | 5. Confusion Matrix |
| ⑪ | Confusion Matrix (이미지) / Normalized (이미지) / 오류 패턴 (mermaid) | — |
| ⑫ | High Confidence Mistakes (이미지) / 개선 방향 표 | — |
| ⑬ | — | 6. 심층 분석 & 7. 개선 방향 & 8. 최종 요약 |
| ⑭ | Day 3 전체 요약 / 체크리스트 | — |

---

### Day 3-5: 고급 기법 & 최종 최적화 (선택)

> 슬라이드: `day3_5-slide-고급 기법 & 최종 최적화.md`  
> 노트북: `lectures/day3/start/day3_5-advanced_techniques-start.ipynb`  
> *Day 3-4까지 완료하면 핵심 목표 달성. 시간이 남는 경우에만 진행.*

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 1. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | 현재 상황 & 목표 / 고급 기법 Overview (mermaid) | — |
| ③ | Data Augmentation 필요성 (mermaid) / 적합·부적합 예시 / Augmentation 이미지 | — |
| ④ | — | 2. Data Augmentation (🔥 rotation_range 등 파라미터) |
| ⑤ | Ensemble 효과 (mermaid) / 3가지 방법 / Ensemble 결과 (이미지) | — |
| ⑥ | — | 3. Model Ensemble (🔥 np.mean Simple Average) |
| ⑦ | TTA란? (mermaid) / TTA 횟수별 성능 (이미지) | — |
| ⑧ | — | 4. TTA 구현 (🔥 np.mean(predictions, axis=0)) |
| ⑨ | 최종 전략: Weighted Ensemble + TTA | — |
| ⑩ | — | 5. 최종 제출 & 6. 최종 성능 분석 (🔥 run_name) |
| ⑪ | Day 3 전체 성능 여정 (이미지) / 핵심 교훈 (mermaid) / 체크리스트 | — |

---

### Day 4-1: COVID-19 흉부 X-ray EDA

> 슬라이드: `day4_1-slide-COVID-19 흉부 X-ray EDA.md`  
> 노트북: `lectures/day4/start/day4_1-covid_eda-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 0. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | 데이터셋 소개 (4클래스 표) / Class Imbalance (이미지+mermaid) | — |
| ③ | X-ray 소견 비교 (mermaid) / Grayscale→RGB / tf.data 파이프라인 | — |
| ④ | — | 1. 데이터 다운로드 & 2. EDA & 3. 샘플 이미지 & 4. 픽셀 분포 |
| ⑤ | Recall의 중요성 (mermaid) / Balanced Accuracy | — |
| ⑥ | Baseline CNN 아키텍처 (mermaid) | — |
| ⑦ | — | 5. 데이터 전처리 & 7. Baseline CNN (🔥 train_test_split, build_baseline_cnn) |
| ⑧ | — | 8. Baseline 모델 학습 (🔥 run_name) |
| ⑨ | 학습 곡선 (이미지) | — |
| ⑩ | — | 9. 모델 평가 |
| ⑪ | Confusion Matrix (이미지) / 결과 분석 / 의료 AI 윤리 / 체크리스트 | — |

---

### Day 4-2: Transfer Learning with ResNet50

> 슬라이드: `day4_2-slide-Transfer Learning with ResNet50.md`
> 노트북: `lectures/day4/start/day4_2-transfer_learning-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 0. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | Day 4-1 결과 복기 / Transfer Learning 아이디어 (mermaid) / Feature Extraction vs Fine-tuning 표 / 2단계 전략 (mermaid) | — |
| ③ | ResNet50 Skip Connection (mermaid) / 전체 구조 (mermaid) / 의료 이미지 Augmentation 주의사항 | — |
| ④ | — | 1. 데이터 로드 & 2. Data Augmentation |
| ⑤ | Phase 1: Feature Extraction 개념 | — |
| ⑥ | — | 3. ResNet50 모델 구축 (🔥 Custom Classifier: Dense + BN + Dropout) |
| ⑦ | — | 4. Phase 1 학습 (🔥 run_name Phase 1) |
| ⑧ | Phase 2: Fine-tuning 개념 / LR 이유 / ReduceLROnPlateau | — |
| ⑨ | — | 5. Phase 2 Fine-tuning (🔥 해제 레이어 수 + run_name Phase 2) |
| ⑩ | — | 6. 성능 평가 & 7. Baseline vs ResNet50 비교 |
| ⑪ | 학습 곡선 (이미지) / Confusion Matrix (이미지) / 비교 (이미지) / COVID Recall 하락 분석 / 체크리스트 | — |

---

### Day 4-3: Class Imbalance 처리

> 슬라이드: `day4_3-slide-Class Imbalance 처리.md`
> 노트북: `lectures/day4/start/day4_3-class_imbalance-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 0. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | Day 4-2 결과 복기 / Class Imbalance 원인 (mermaid) | — |
| ③ | 해결 전략 Overview (mermaid) | — |
| ④ | Class Weights 원리 / 계산 수식 | — |
| ⑤ | — | 1. 데이터 로드 & 2. Class Weights 계산 (🔥 compute_class_weight) |
| ⑥ | Class Weights 적용 코드 | — |
| ⑦ | — | 3. Dataset 생성 & 4. 모델 구축 |
| ⑧ | — | 5. Experiment 1: Class Weights 학습 (🔥 run_name, class_weight) |
| ⑨ | Resampling 개념 (mermaid + 표) | — |
| ⑩ | Cross Entropy의 문제 / Focal Loss 수식 (mermaid) | — |
| ⑪ | Focal Loss 구현 코드 | — |
| ⑫ | — | 6. Experiment 2: Focal Loss 구현 & 학습 (🔥 focal_term, loss, run_name) |
| ⑬ | — | 7. 성능 비교 & 8. Best 모델 선정 |
| ⑭ | 실험 결과 수치 / 결과 이미지 / Confusion Matrix (이미지) | — |
| ⑮ | 의료 AI 비용 비대칭 (mermaid) / 체크리스트 | — |

### Day 4-4: Grad-CAM & 최종 평가

> 슬라이드: `day4_4-slide-Grad-CAM & 최종 평가.md`
> 노트북: `lectures/day4/start/day4_4-gradcam_final-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 0. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | Day 4 성능 개선 흐름 (mermaid + 표) | — |
| ③ | — | 1. MLflow에서 Best 모델 로드 |
| ④ | Grad-CAM 필요성 / 알고리즘 (mermaid) / 핵심 수식 | — |
| ⑤ | Grad-CAM 구현 코드 (make_gradcam_heatmap) | — |
| ⑥ | — | 2. Grad-CAM 구현 (🔥 pooled_grads, heatmap weighted sum, heatmap ReLU+normalize) |
| ⑦ | — | 3. 클래스별 Grad-CAM 시각화 (overlay_heatmap) |
| ⑧ | 결과 이미지 / COVID 소견 대조 (mermaid) | — |
| ⑨ | — | 4. FP/FN 분석 |
| ⑩ | COVID FN Grad-CAM 이미지 | — |
| ⑪ | ROC-AUC 코드 / 결과 이미지 | — |
| ⑫ | PR Curve 코드 / 결과 이미지 | — |
| ⑬ | — | 5-7. ROC-AUC & PR Curve & Day 4 최종 종합 평가 |
| ⑭ | 성능 여정 이미지 / 최종 수치 / 기술 스택 (mermaid) / 임상 적용 가능성 / 체크리스트 | — |

---

### Day 5-1: HaGRID 손 제스처 데이터 EDA

> 슬라이드: `day5_1-slide-HaGRID 손 제스처 데이터 EDA.md`
> 노트북: `lectures/day5/start/day5_1-hagrid_eda-start.ipynb`

| 단계 | 슬라이드 | 노트북 |
|:---:|:---|:---|
| ① | — | 0. 환경 설정 (🔥 repo_owner, repo_name) |
| ② | HaGRID 데이터셋 소개 (19개 클래스 표) / Day 4 vs Day 5 비교 (mermaid + 표) / 경량 모델 필요성 (mermaid) | — |
| ③ | — | 1. HaGRID 데이터 다운로드 (Kaggle API) |
| ④ | — | 2. 데이터 구조 탐색 (폴더 확인, 클래스별 이미지 수) |
| ⑤ | 클래스 분포 이미지 / 샘플 이미지 | — |
| ⑥ | — | 3. 샘플 이미지 시각화 (5×4 그리드) |
| ⑦ | tf.data 파이프라인 코드 (JPEG+RGB 설명) | — |
| ⑧ | — | 4. Train/Val/Test Split (🔥 train_test_split 70/15/15) |
| ⑨ | — | 5. tf.data.Dataset 생성 |
| ⑩ | Baseline CNN 아키텍처 (mermaid) | — |
| ⑪ | — | 6. Baseline CNN 구현 (🔥 3개 Conv Block) |
| ⑫ | — | 7. Baseline 모델 학습 (🔥 run_name) |
| ⑬ | — | 8. 모델 평가 (학습 곡선 / Confusion Matrix) |
| ⑭ | 학습 곡선 이미지 / Confusion Matrix 이미지 | — |
| ⑮ | Baseline 결과 / Day 5-2 예고 (mermaid) / 체크리스트 | — |

---

## 옵션 설명

- `--name slidev-m1` : 컨테이너 이름
- `--rm` : 종료 시 컨테이너 자동 삭제
- `-it` : 터미널 인터랙티브
- `-v "$(pwd):/slidev"` : 현재 디렉터리를 컨테이너 `/slidev`에 마운트
- `-p 3030:3030` : 호스트 3030 포트로 접속
- `--remote` : Docker 내부에서 실행할 때 외부(호스트)에서 접속 가능하도록 바인딩
