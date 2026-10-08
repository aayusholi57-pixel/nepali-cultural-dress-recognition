# 🇳🇵 Nepali Cultural Dress Recognition

> **An end-to-end computer vision system for recognizing Nepali cultural dresses and ornaments using a custom ResNet50 V4 architecture, two-stage training, FastAPI, Docker, Amazon S3, and Amazon EC2.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50%20Custom%20V4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-232F3E?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)

---

## 1. Project Overview

**Nepali Cultural Dress Recognition** is a 24-class image-classification system for recognizing Nepali cultural dresses, ornaments, and related accessories.

The current production workflow uses **ResNet50 Custom V4**. The system combines a pretrained ResNet50 feature extractor with custom convolutional and linear layers, then serves the trained model through FastAPI.

The project covers the complete machine-learning lifecycle:

```text
Dataset
  ↓
Dataset validation and preparation
  ↓
Image preprocessing and augmentation
  ↓
Pretrained ResNet50 backbone
  ↓
Custom Conv2D layers
  ↓
Custom Linear classification head
  ↓
Two-stage V4 training
  ↓
Validation and test evaluation
  ↓
Best V4 checkpoint
  ↓
Amazon S3
  ↓
AWS EC2
  ↓
Docker
  ↓
FastAPI
  ↓
Image prediction
```

---

## 2. Current Status

| Item | Current value |
|---|---|
| Current model | **ResNet50 Custom V4** |
| Model artifact | `best_resnet50_custom_v4.pth` |
| Artifact version | **4.0** |
| Classes | **24** |
| Total training epochs | **10** |
| Best checkpoint | **Overall Epoch 8 / Phase 2 Epoch 3** |
| Validation accuracy | **87.63%** |
| Validation Macro F1 | **63.10%** |
| Test accuracy | **75.00%** |
| Test Macro F1 | **64.01%** |
| Test Weighted F1 | **73.45%** |
| Model storage | Amazon S3 |
| Compute | AWS EC2 |
| Containerization | Docker |
| API | FastAPI |
| Inference device | CPU |
| Confidence threshold | **0.60** |

> **V4 is the current model used by the deployment workflow. V3 is retained only as a previous benchmark for experiment history.**

---

## 3. Why V4?

The V4 architecture follows the mentor-guided requirement to keep **ResNet50 as the pretrained feature extractor** while adding custom neural-network layers for the final classification stage.

Instead of using the original ResNet50 classification head, V4 adds:

- two custom Conv2D layers
- global average pooling
- two custom Linear layers
- ReLU activations
- Dropout

This makes the architecture a **ResNet50-based custom classification system**, rather than a plain ResNet50 classifier.

---

# 4. V4 Architecture

```text
Input Image
224 × 224 × 3
      │
      ▼
Image Preprocessing
Resize + Tensor + ImageNet Normalization
      │
      ▼
PRETRAINED RESNET50 BACKBONE
      │
      ├── Conv1
      ├── BatchNorm
      ├── ReLU
      ├── MaxPool
      ├── Layer1
      ├── Layer2
      ├── Layer3
      └── Layer4
      │
      ▼
Feature Map
2048 × 7 × 7
      │
      ▼
Custom Conv2D
2048 → 512
Kernel 3 × 3
Padding 1
      │
      ▼
ReLU
      │
      ▼
Custom Conv2D
512 → 256
Kernel 3 × 3
Padding 1
      │
      ▼
ReLU
      │
      ▼
Global Average Pooling
256 × 7 × 7 → 256 × 1 × 1
      │
      ▼
Flatten
256
      │
      ▼
Custom Linear
256 → 128
      │
      ▼
ReLU
      │
      ▼
Dropout
p = 0.2
      │
      ▼
Custom Linear
128 → 24
      │
      ▼
24 Class Logits
      │
      ▼
Softmax
      │
      ▼
Top-1 / Top-3 Predictions
```

### Architecture summary

| Component | Configuration |
|---|---|
| Input | 224 × 224 × 3 |
| Backbone | ImageNet-pretrained ResNet50 |
| Backbone output | 2048 × 7 × 7 |
| Custom Conv1 | 2048 → 512, 3 × 3, padding 1 |
| Custom Conv2 | 512 → 256, 3 × 3, padding 1 |
| Activation | ReLU |
| Pooling | Adaptive Average Pooling |
| Custom Linear1 | 256 → 128 |
| Dropout | 0.2 |
| Custom Linear2 | 128 → 24 |
| Output | 24 class logits |

Implementation:

```text
src/v4_model.py
```

---

# 5. What Does 2048 × 7 × 7 Mean?

After the ResNet50 backbone, the image becomes a feature representation of:

```text
2048 channels × 7 × 7 spatial resolution
```

This does **not** mean 2048 separate images.

It means the backbone produces **2048 feature channels**, where each channel contains a 7 × 7 spatial feature map.

Total feature values:

```text
2048 × 7 × 7 = 100,352
```

The custom Conv2D layers reduce the representation:

```text
2048 channels
      ↓
512 channels
      ↓
256 channels
```

The 3 × 3 convolutions use padding 1, so the 7 × 7 spatial size is preserved before global average pooling.

---

# 6. Training Strategy

V4 uses **two-stage training with 10 total epochs**.

```text
                 V4 TRAINING
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
       PHASE 1                 PHASE 2
      Epochs 1–5              Epochs 6–10
          │                       │
          │                       │
   ResNet50 frozen        Layers 1–3 frozen
   Custom layers train    Layer 4 trainable
                          Custom layers train
          │                       │
          └───────────┬───────────┘
                      ▼
             Validation each epoch
                      │
                      ▼
          Best checkpoint by Macro F1
                      │
                      ▼
              Epoch 8 / 10
          Phase 2, Epoch 3
```

### Phase 1 — Custom head training

| Component | State |
|---|---|
| ResNet50 backbone | Frozen |
| ResNet50 Layer 1 | Frozen |
| ResNet50 Layer 2 | Frozen |
| ResNet50 Layer 3 | Frozen |
| ResNet50 Layer 4 | Frozen |
| Custom Conv1 | Trainable |
| Custom Conv2 | Trainable |
| Custom Linear1 | Trainable |
| Custom Linear2 | Trainable |
| Epochs | 5 |
| Learning rate | 1e-3 |

### Phase 2 — Layer 4 fine-tuning

| Component | State |
|---|---|
| ResNet50 Layers 1–3 | Frozen |
| ResNet50 Layer 4 | Trainable |
| Custom Conv1 | Trainable |
| Custom Conv2 | Trainable |
| Custom Linear1 | Trainable |
| Custom Linear2 | Trainable |
| Epochs | 5 |
| Layer 4 learning rate | 1e-5 |
| Custom-layer learning rate | 1e-3 |

---

# 7. Training Configuration

| Setting | V4 value |
|---|---|
| Architecture | ResNet50 Custom V4 |
| Initialization | ImageNet pretrained |
| Input size | 224 × 224 |
| Batch size | 8 |
| Phase 1 | 5 epochs |
| Phase 2 | 5 epochs |
| Total | **10 epochs** |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Loss | Weighted Cross Entropy |
| Class weighting | Inverse square-root |
| Dropout | 0.2 |
| Random seed | 42 |
| Training device | CPU |
| Selection metric | Validation Macro F1 |

Training implementation:

```text
src/train_v4.py
```

Run:

```bash
python -m src.train_v4
```

---

# 8. Data Augmentation

Training preprocessing:

```text
Input image
   ↓
RandomResizedCrop(224, scale=0.8–1.0)
   ↓
RandomHorizontalFlip
   ↓
RandomRotation(10°)
   ↓
ColorJitter
   ↓
ToTensor
   ↓
ImageNet normalization
```

Validation and inference preprocessing:

```text
Input image
   ↓
Resize 224 × 224
   ↓
ToTensor
   ↓
ImageNet normalization
```

ImageNet normalization:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

# 9. Dataset

The project uses the PyTorch `ImageFolder` format.

```text
data/
├── train/
├── val/
└── test/
```

Dataset sizes:

| Split | Images | Classes represented |
|---|---:|---:|
| Train | **1,586** | 24 |
| Validation | **299** | 24 |
| Test | **192** | 23 of 24 |

The test split does not contain `naugedi`.

The dataset preparation code validates:

- required dataset splits
- class-folder consistency
- supported image extensions
- train/validation class mapping
- image counts
- empty classes
- class consistency

Implementation:

```text
src/dataset_prep.py
```

---

# 10. Supported Classes

The V4 model predicts 24 classes:

1. Ali Band
2. Bijaith
3. Chura
4. Haku Tapli
5. Hansuli
6. Hāku patāsi
7. Mangiya
8. Nyapu Shikha
9. Sirbandi
10. Tapalan
11. cholo
12. chyapte_sun
13. dhungri
14. dori
15. gunyo
16. kanthula
17. naugedi
18. patuki
19. potey
20. sirful
21. teekma
22. tharu_broad_wrist_band
23. tharu_head_cloth
24. tharu_waist_chain

---

# 11. Class Imbalance

V4 uses inverse-square-root class weighting:

```text
weight(class) = 1 / sqrt(class_count)
```

The weights are normalized so their mean is approximately 1.

This provides a softer correction than direct inverse-frequency weighting and reduces the risk of very rare classes dominating the training loss.

---

# 12. Model Selection

The best V4 checkpoint is selected using **validation Macro F1**, not validation accuracy alone.

Why Macro F1?

Because this is a multi-class problem with class imbalance. Macro F1 gives every class equal importance.

The selected checkpoint is:

```text
Overall Epoch : 8 / 10
Phase         : Phase 2
Phase Epoch   : 3
Val Accuracy  : 87.63%
Val Macro F1  : 63.10%
```

Checkpoint:

```text
models/resnet50_custom_v4/best_resnet50_custom_v4.pth
```

---

# 13. V4 Evaluation Results

## Validation

| Metric | Result |
|---|---:|
| Accuracy | **87.63%** |
| Macro F1 | **63.10%** |

## Test

| Metric | Result |
|---|---:|
| Accuracy | **75.00%** |
| Macro F1 | **64.01%** |
| Weighted F1 | **73.45%** |
| Test images | **192** |
| Classes represented | **23 / 24** |

Evaluation implementation:

```text
src/evaluate_v4.py
```

Run:

```bash
python -m src.evaluate_v4
```

---

# 14. Test-Set Caveat

The test set contains examples from **23 of the 24 classes**.

The missing class is:

```text
naugedi
```

Therefore:

- the reported overall metrics describe the available 192 test images
- `naugedi` cannot be evaluated from the current test split
- per-class metrics for absent classes should not be interpreted as measured test performance

This limitation is intentionally documented rather than hidden.

---

# 15. Strong and Difficult Classes

The current V4 evaluation shows strong performance on several classes, including:

| Class | F1 |
|---|---:|
| Haku Tapli | **1.00** |
| Nyapu Shikha | **1.00** |
| tharu_broad_wrist_band | **0.97** |
| Hansuli | **0.96** |
| Hāku patāsi | **0.92** |
| Mangiya | **0.91** |
| Tapalan | **0.88** |

More challenging classes include:

- `chyapte_sun`
- `dhungri`
- `patuki`
- `dori`
- `potey`

Performance can vary with image quality, pose, background, occlusion, and visual similarity between cultural items.

---

# 16. V3 Benchmark

V3 is retained as a **previous benchmark**, not as the deployed model.

| Metric | V3 |
|---|---:|
| Test Accuracy | **82.29%** |
| Test Macro F1 | **70.90%** |
| Test Weighted F1 | **82.73%** |

V4 has lower test metrics than V3, but V4 was selected for the current workflow because it implements the mentor-directed architecture with custom convolutional and linear layers.

This distinction is important:

```text
V3
↓
Previous benchmark

V4
↓
Current model + deployment architecture
```

---

# 17. V4 vs V3

| Area | V3 | V4 |
|---|---|---|
| Backbone | ResNet50 | ResNet50 |
| Custom Conv2D | No | **Yes** |
| Custom Linear head | Yes | **Yes** |
| Training | 5 + 15 epochs | **5 + 5 epochs** |
| Test Accuracy | 82.29% | **75.00%** |
| Test Macro F1 | 70.90% | **64.01%** |
| Current deployment | Previous benchmark | **Current** |

A newer architecture is not automatically more accurate. The project keeps the benchmark history visible while using V4 as the current engineering/deployment implementation.

---

# 18. Production Inference Architecture

```text
                    User / Client
                         │
                         ▼
                    FastAPI API
                         │
                         ▼
                  PredictorService
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Image validation       Image preprocessing
              │                     │
              └──────────┬──────────┘
                         ▼
                ResNet50 Custom V4
                         │
                         ▼
                 24 class logits
                         │
                         ▼
                    Softmax
                         │
                         ▼
                 Top-3 predictions
                         │
                         ▼
              Confidence check 0.60
                         │
                         ▼
                    API response
```

---

# 19. Cloud Deployment Architecture

```text
                         AWS
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        Amazon S3                 Amazon EC2
        Model Artifact                 │
              │                        ▼
              │                   Docker
              │                        │
              └───────────────────────►│
                                       ▼
                                   FastAPI
                                       │
                                       ▼
                              ResNet50 Custom V4
                                       │
                                       ▼
                                  Port 9000
```

Current model artifact:

```text
S3 bucket:
nepali-cultural-dress-ai-696822062401-ap-southeast-2-an

S3 key:
models/resnet50_custom_v4/best_resnet50_custom_v4.pth
```

---

# 20. Complete End-to-End Workflow

```text
Dataset
   ↓
Dataset Preparation
   ↓
Train / Validation / Test Split
   ↓
Image Preprocessing
   ↓
Pretrained ResNet50 Backbone
   ↓
2048 × 7 × 7 Feature Map
   ↓
Custom Conv2D
2048 → 512
   ↓
Custom Conv2D
512 → 256
   ↓
Global Average Pooling
   ↓
Custom Linear
256 → 128
   ↓
Dropout 0.2
   ↓
Custom Linear
128 → 24
   ↓
Phase 1 Training
   ↓
Phase 2 Fine-Tuning
   ↓
Validation
   ↓
Best Checkpoint by Macro F1
   ↓
V4 Checkpoint
   ↓
Amazon S3
   ↓
AWS EC2
   ↓
Docker
   ↓
FastAPI
   ↓
User Uploads Image
   ↓
Validation + Preprocessing
   ↓
V4 Inference
   ↓
Top-3 Predictions
   ↓
Confidence / Uncertainty Check
   ↓
API Response
```

---

# 21. FastAPI

The API is implemented using FastAPI.

Application:

```text
api/main.py
```

Inference service:

```text
api/predictor.py
```

Schemas:

```text
api/schemas.py
```

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Service and model health |
| `/model-info` | GET | Model metadata |
| `/predict` | POST | Image classification |
| `/docs` | GET | Swagger UI |
| `/openapi.json` | GET | OpenAPI schema |

---

# 22. Prediction API

The prediction endpoint accepts:

- JPEG
- PNG
- WEBP

The service:

1. validates the MIME type
2. reads the uploaded file
3. checks that the file is not empty
4. enforces a 5 MB image limit
5. decodes the image with Pillow
6. converts it to RGB
7. resizes it to 224 × 224
8. applies ImageNet normalization
9. runs V4 inference
10. returns the top-3 predictions
11. applies the confidence threshold

---

# 23. Confidence / Uncertainty

The service uses:

```text
Confidence threshold = 0.60
```

If the highest predicted confidence is below 0.60:

```text
status = "uncertain"
```

If it is at or above 0.60:

```text
status = "success"
```

This is a **confidence-based uncertainty indicator**.

It is **not a formal or calibrated Out-of-Distribution detector** and should not be interpreted as proof that an image belongs outside the training distribution.

---

# 24. API Response Concept

A successful prediction returns information such as:

```json
{
  "filename": "sample.jpg",
  "status": "success",
  "top_prediction": "Hansuli",
  "confidence": 94.21,
  "top_k": [
    {
      "class_name": "Hansuli",
      "confidence": 94.21
    },
    {
      "class_name": "Mangiya",
      "confidence": 3.42
    },
    {
      "class_name": "Chura",
      "confidence": 1.13
    }
  ],
  "is_out_of_distribution": false
}
```

The field name `is_out_of_distribution` is retained for API compatibility, but the decision is currently based on the **0.60 confidence threshold**, not a formal OOD model.

---

# 25. Model Metadata API

`/model-info` exposes the deployed model metadata, including:

- architecture
- number of classes
- supported classes
- inference device
- confidence threshold
- artifact version
- checkpoint epoch
- validation accuracy
- validation Macro F1

The expected V4 metadata is:

```text
architecture:
ResNet50 Custom V4

artifact_version:
4.0

epoch:
3  (Phase 2 epoch; overall Epoch 8)

val_accuracy:
0.8762541806020067

val_macro_f1:
0.6310008087372548
```

---

# 26. Docker

The inference service is containerized with Docker.

Build:

```bash
docker build -t nepali-dress-api .
```

Run with Compose:

```bash
docker compose up --build
```

Port mapping:

```text
Host:      9000
Container: 8000
```

The container obtains the V4 model from Amazon S3 at startup.

---

# 27. AWS EC2 Deployment

The deployed service runs on Amazon EC2 using Ubuntu and Docker.

Deployment environment:

| Component | Value |
|---|---|
| Cloud | AWS |
| Compute | EC2 |
| Region | ap-southeast-2 |
| Container | Docker |
| API | FastAPI |
| Model storage | Amazon S3 |
| Inference | CPU |
| Host port | 9000 |
| Container port | 8000 |
| Model | ResNet50 Custom V4 |

The API downloads the V4 checkpoint from S3 during application startup.

If the model cannot be loaded, the service does not report a healthy model-loaded state.

---

# 28. CI/CD Deployment

GitHub Actions is configured to deploy the application to EC2 when changes are pushed to `main`.

The deployment process:

```text
Git push
   ↓
GitHub Actions
   ↓
SSH to EC2
   ↓
Clone latest main
   ↓
Verify V4 deployment files
   ↓
Build Docker image
   ↓
Start container
   ↓
Wait for /health
   ↓
Verify model_loaded=true
   ↓
Verify /model-info
   ↓
Final health verification
   ↓
Deployment successful
```

Workflow:

```text
.github/workflows/deploy.yml
```

The deployment is **health-gated**: a successful Docker build alone is not considered a successful deployment.

---

# 29. AWS Credentials

The application uses the standard boto3 credential chain.

For EC2, an IAM instance role is preferred over hard-coded credentials.

The application does not store AWS credentials in Python source code.

Required runtime configuration includes:

```text
AWS_REGION
AWS_DEFAULT_REGION
S3_BUCKET
S3_MODEL_KEY
```

Current production model key:

```text
models/resnet50_custom_v4/best_resnet50_custom_v4.pth
```

---

# 30. Security and Reliability

Current engineering practices include:

- no AWS credentials hard-coded in source
- S3 model storage separated from application code
- runtime environment variables for deployment configuration
- image MIME-type validation
- image decoding validation
- 5 MB upload limit
- model-loaded health verification
- Dockerized runtime
- EC2 IAM credential-chain compatibility
- non-root container runtime
- health-gated deployment

For production hardening, least-privilege IAM permissions should be used for the EC2 role.

---

# 31. Repository Structure

```text
nepali-cultural-dress-recognition/
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── src/
│   ├── __init__.py
│   ├── dataset_prep.py
│   ├── evaluate.py
│   ├── evaluate_v4.py
│   ├── test_v4_model.py
│   ├── train_v4.py
│   ├── v4_model.py
│   ├── model.py
│   ├── train.py
│   ├── hybrid_model.py
│   ├── train_hybrid.py
│   ├── evaluate_hybrid.py
│   └── utils.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .dockerignore
├── .gitignore
└── README.md
```

V3 and Hybrid V1 source files remain in the repository as experiment history; **they are not used by the current V4 deployment path**.

---

# 32. Local Setup

Clone:

```bash
git clone https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition.git
cd nepali-cultural-dress-recognition
```

Create environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate dataset:

```bash
python -m src.dataset_prep
```

Train V4:

```bash
python -m src.train_v4
```

Evaluate V4:

```bash
python -m src.evaluate_v4
```

Run architecture tests:

```bash
python -m src.test_v4_model
```

Start API:

```bash
uvicorn api.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# 33. Experiment History

The project evolved through multiple architecture experiments.

```text
V3
│
├── Previous ResNet50 benchmark
├── Test Accuracy: 82.29%
└── Test Macro F1: 70.90%
        │
        ▼
Hybrid V1
│
├── Reduced ResNet50 feature extractor
├── Custom CNN
└── Separate research experiment
        │
        ▼
V4
│
├── Full ResNet50 feature extractor
├── Custom Conv2D layers
├── Custom Linear layers
├── 10 total epochs
└── Current deployment model
```

The project intentionally keeps historical experiments separate from the current production path.

---

# 34. Limitations and Future Work

### Current limitations

- the dataset is relatively limited for a 24-class cultural recognition problem
- the test set lacks `naugedi`
- some classes have visually similar features
- performance depends on image quality, pose, lighting, and background
- CPU inference is slower than GPU inference
- the confidence threshold is not calibrated
- the uncertainty mechanism is not a formal OOD detector
- the model is designed for a fixed 24-class label space
- AWS/S3 availability is required during production startup

### Future improvements

- [ ] Add test examples for all 24 classes
- [ ] Collect a larger and more diverse dataset
- [ ] Calibrate confidence scores
- [ ] Add a dedicated OOD detection method
- [ ] Add latency and throughput benchmarks
- [ ] Add model versioning
- [ ] Add experiment tracking
- [ ] Add dataset versioning
- [ ] Add model monitoring
- [ ] Optimize CPU inference
- [ ] Add GPU deployment where appropriate
- [ ] Add HTTPS and reverse proxy
- [ ] Add infrastructure-as-code
- [ ] Add deployment rollback
- [ ] Add API authentication and rate limiting

---

# 35. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision + Pillow |
| Model | ResNet50 Custom V4 |
| Metrics | scikit-learn |
| API | FastAPI |
| ASGI server | Uvicorn |
| Validation | Pydantic |
| AWS SDK | boto3 |
| Model storage | Amazon S3 |
| Compute | Amazon EC2 |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Dataset | PyTorch ImageFolder |

---

# 36. Why This Project Matters

This project demonstrates an end-to-end ML engineering workflow:

```text
Dataset Engineering
       ↓
Computer Vision
       ↓
Transfer Learning
       ↓
Custom Neural-Network Architecture
       ↓
Two-Stage Fine-Tuning
       ↓
Model Evaluation
       ↓
Model Packaging
       ↓
FastAPI
       ↓
Docker
       ↓
Amazon S3
       ↓
Amazon EC2
       ↓
GitHub Actions
       ↓
Health-Gated Deployment
```

It demonstrates practical experience across:

- Computer Vision
- Deep Learning
- Transfer Learning
- PyTorch
- Model evaluation
- FastAPI
- Docker
- AWS
- S3
- EC2
- CI/CD
- Production-oriented ML engineering

---

# 37. Project Links

Repository:

https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition

When the EC2 service is running:

```text
Swagger:
http://52.65.181.90:9000/docs

Health:
http://52.65.181.90:9000/health

Model information:
http://52.65.181.90:9000/model-info

OpenAPI:
http://52.65.181.90:9000/openapi.json
```

> Availability of the EC2 endpoint depends on the current instance, networking, and container state.

---

# 38. Author

## Aayush Oli

AI/ML Engineer in training focused on:

- Computer Vision
- Deep Learning
- Machine Learning Engineering
- FastAPI
- Docker
- AWS
- Production AI systems

---

# 39. License

A formal open-source license has not yet been selected for this repository.

If the project is intended for public reuse, add an appropriate license before presenting it as an open-source project.

---

<p align="center">
  <b>🇳🇵 Built to digitally recognize and preserve Nepal's cultural heritage.</b>
</p>
