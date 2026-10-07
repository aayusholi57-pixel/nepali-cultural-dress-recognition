# 🇳🇵 Nepali Cultural Dress Recognition

> **An end-to-end computer vision system for recognizing Nepali cultural dresses and ornaments — from dataset preparation and transfer learning to a production FastAPI service, Docker, Amazon S3, Amazon EC2, and automated CI/CD.**

[![CI/CD](https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition/actions/workflows/deploy.yml/badge.svg)](https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition/actions/workflows/deploy.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-232F3E?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/License-To%20be%20added-lightgrey)](#license)

---

## 📌 What This Project Does

**Nepali Cultural Dress Recognition** is a production-oriented image-classification system that identifies Nepali cultural dresses and ornaments from images.

The project is intentionally built as an **end-to-end ML engineering system**, not only as a notebook or training script:

```text
Dataset
   ↓
Validation & Preparation
   ↓
Augmentation + ImageNet Normalization
   ↓
ResNet50 Transfer Learning
   ↓
Two-Stage Fine-Tuning
   ↓
Validation / Test Evaluation
   ↓
Model Checkpoint
   ↓
Amazon S3
   ↓
FastAPI Inference Service
   ↓
Docker Container
   ↓
Amazon EC2
   ↓
GitHub Actions CI/CD
```

The trained model recognizes **24 target classes** and returns ranked predictions with confidence scores. If the highest confidence falls below the configured threshold, the API reports the result as **uncertain** rather than presenting it as a confident classification.

---

## ✨ Highlights

- 🧠 **ResNet50 transfer learning** with ImageNet initialization
- 🎯 **Two-stage fine-tuning** for controlled domain adaptation
- ⚖️ **Class-imbalance-aware training** using inverse-square-root class weighting
- 🖼️ **Production-consistent preprocessing** at 224×224
- 📊 **Accuracy, Macro F1, Weighted F1, classification report, and confusion matrix**
- 🏷️ **24-class cultural dress / ornament classification**
- 🔎 **Top-3 inference results**
- 🚦 **Confidence-based uncertainty detection**
- ☁️ **Amazon S3 model artifact storage**
- 🐳 **Dockerized inference**
- 🚀 **Amazon EC2 deployment**
- 🔄 **GitHub Actions automated deployment**
- ❤️ **Health endpoint with model-loaded verification**
- ℹ️ **Model metadata endpoint**
- 🔐 **AWS credential-chain support / EC2 IAM role compatibility**
- 🛡️ **Non-root container runtime**
- 📦 **Self-describing model checkpoint metadata**
- 🧪 **Dataset and class-consistency validation**

---

# 🏗️ System Architecture

```text
                              ┌──────────────────────┐
                              │   User / Client      │
                              │  Image Upload        │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │      FastAPI         │
                              │                      │
                              │  /                  │
                              │  /health            │
                              │  /model-info        │
                              │  /predict            │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │   PredictorService   │
                              │                      │
                              │ Validation           │
                              │ RGB conversion       │
                              │ Preprocessing        │
                              │ Inference            │
                              │ Top-K ranking        │
                              │ Confidence decision  │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │      ResNet50        │
                              │   24-class model     │
                              └──────────▲───────────┘
                                         │
                               model artifact
                                         │
                              ┌──────────┴───────────┐
                              │      Amazon S3        │
                              │                      │
                              │ best_resnet50.pth   │
                              └──────────────────────┘


        ┌───────────────────── CI/CD ─────────────────────┐
        │                                                   │
        │  GitHub → GitHub Actions → SSH → EC2 → Docker   │
        │                                                   │
        └───────────────────────────────────────────────────┘
```

### Deployment flow

```text
git push
   │
   ▼
GitHub Actions
   │
   ├── checkout main
   ├── validate deployment files
   ├── build Docker image
   ├── start container
   ├── verify /health
   ├── verify model_loaded=true
   └── verify /model-info
             │
             ▼
         Amazon EC2
             │
             ▼
       FastAPI container
             │
             ▼
        Amazon S3 model
```

---

# 🤖 Machine Learning

## Model

The project uses **ResNet50** with ImageNet pretrained weights.

The original ResNet50 classification layer is replaced with a custom classification head:

```text
ResNet50 backbone
      │
      └── 2048 feature representation
                │
                ▼
        Dropout(0.4)
                │
                ▼
          Linear → 512
                │
                ▼
              ReLU
                │
                ▼
          BatchNorm1d(512)
                │
                ▼
          Dropout(0.2)
                │
                ▼
          Linear → 24
                │
                ▼
        Class probabilities
```

The model is implemented in:

```text
src/model.py
```

---

# 🎯 Two-Stage Fine-Tuning

Rather than immediately updating the entire pretrained network, training is divided into two controlled phases.

## Phase 1 — Train the classification head

The ResNet50 backbone is frozen.

```text
Backbone      → Frozen
Custom head   → Trainable
Learning rate → 1e-3
Epochs        → 5
```

This lets the new classifier learn the target label space before modifying the pretrained feature extractor.

## Phase 2 — Domain adaptation

ResNet50 `layer4` and the classification head are unfrozen.

```text
Early backbone → Frozen
layer4         → Trainable
Classifier     → Trainable

layer4 LR      → 1e-5
classifier LR  → 1e-4
Epochs         → 15
```

This provides controlled adaptation to Nepali cultural clothing imagery while preserving most of the pretrained representation.

---

# 📊 Training Configuration

| Component | Configuration |
|---|---|
| Architecture | ResNet50 |
| Initialization | ImageNet pretrained |
| Number of classes | 24 |
| Input size | 224 × 224 |
| Batch size | 8 |
| Phase 1 | 5 epochs |
| Phase 2 | 15 epochs |
| Head learning rate | 1e-3 |
| Layer4 learning rate | 1e-5 |
| Fine-tuned head learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Loss | Weighted Cross Entropy |
| Class weighting | Inverse square root |
| Selection metric | Validation Macro F1 |
| Random seed | 42 |
| Training target | CPU-compatible |
| Total training schedule | **20 epochs (5 + 15)** |
| Best V3 checkpoint | **Phase 2, epoch 7** |

Run training with:

```bash
python -m src.train
```

---

# ⚖️ Class Imbalance Strategy

The training set is not perfectly balanced across classes.

Instead of using raw inverse-frequency weighting, the project uses:

```text
weight(class) = 1 / sqrt(class_count)
```

The resulting weights are normalized so their mean is approximately 1.

This is deliberately less aggressive than direct inverse-frequency weighting and helps prevent very small classes from dominating the loss.

---

# 🖼️ Image Pipeline

### Training

```text
Input image
   ↓
RandomResizedCrop(224)
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

### Validation / inference

```text
Input image
   ↓
Resize → 224 × 224
   ↓
ToTensor
   ↓
ImageNet normalization
   ↓
ResNet50
```

Using the same deterministic normalization assumptions between evaluation and inference reduces training/serving preprocessing mismatch.

---

# 🧪 Dataset

The training and evaluation data follow the PyTorch `ImageFolder` convention:

```text
data/
├── train/
│   ├── class_01/
│   ├── class_02/
│   └── ...
├── val/
│   ├── class_01/
│   ├── class_02/
│   └── ...
└── test/
    ├── class_01/
    ├── class_02/
    └── ...
```

Dataset validation is implemented in:

```text
src/dataset_prep.py
```

The pipeline checks important dataset assumptions such as:

- required splits
- class-folder consistency
- supported image extensions
- image counts
- empty classes
- train/validation class mismatches
- reproducible class-index mapping

Run:

```bash
python -m src.dataset_prep
```

---

# 📈 Evaluation

## Current Experiment Snapshot

The latest verified **V3 experiment** used the repository's **20-epoch training schedule** (5 + 15).

### Dataset snapshot

| Split | Images | Classes represented |
|---|---:|---:|
| Train | **1,586** | 24 |
| Validation | **299** | 24 |
| Test | **192** | 23 of 24 |

The test split is missing the `naugedi` class. The evaluation script detects this condition and remaps the remaining test labels back to the original 24-class model indices so that class-index shifting does not corrupt the evaluation.

| Phase | Epochs | Purpose |
|---|---:|---|
| Phase 1 | **5** | Train the custom classification head with the ResNet50 backbone frozen |
| Phase 2 | **15** | Fine-tune ResNet50 `layer4` together with the classification head |
| **Total** | **20** | Complete training schedule |

The V3 best checkpoint was recorded at **Phase 2, epoch 7** — the **12th epoch overall** (5 Phase-1 epochs + 7 Phase-2 epochs).

Evaluation is implemented in:

```text
src/evaluate.py
```

Run:

```bash
python -m src.evaluate
```

The evaluation pipeline generates:

```text
evaluation/
├── metrics.json
└── confusion_matrix.png
```

It reports:

- Accuracy
- Macro F1
- Weighted F1
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix
- Best validation metrics stored in the checkpoint

### Important test-set note

The current test split does not contain examples for every one of the 24 model classes. The evaluation code explicitly detects missing test classes and remaps the remaining ImageFolder labels back to the original 24-class model indices.

This prevents a subtle class-index shift from producing incorrect evaluation results.

Because some classes have no ground-truth test samples, their per-class metrics cannot be interpreted as a real measure of test performance.

### V3 final evaluation

The **V3 production experiment** is the current benchmark for this repository.

| Metric | V3 result |
|---|---:|
| Validation accuracy | **84.62%** |
| Validation Macro F1 | **61.44%** |
| Test accuracy | **82.29%** |
| Test Macro F1 | **70.90%** |
| Test Weighted F1 | **82.73%** |
| Best checkpoint | **Phase 2, epoch 7** |

V3 is the result to use when describing the current model in a portfolio, interview, or deployment discussion. Older experiment results should not be presented as the current benchmark.

> **Evaluation caveat:** the test split contains 192 images across 23 of the 24 model classes; `naugedi` has no test examples. Therefore, the overall test metrics are valid for the available test samples, but the absent class cannot be evaluated from this split.

---

## 🧾 Current V3 Experiment Record

| Item | Verified value |
|---|---|
| Experiment | **V3** |
| Model | ResNet50 transfer learning |
| Classes | **24** |
| Train images | **1,586** |
| Validation images | **299** |
| Test images | **192** |
| Test classes represented | **23 / 24** |
| Missing test class | **naugedi** |
| Phase 1 | **5 epochs** |
| Phase 2 | **15 epochs** |
| Total schedule | **20 epochs** |
| Best checkpoint | **Phase 2, epoch 7** |
| Validation accuracy | **84.62%** |
| Validation Macro F1 | **61.44%** |
| Test accuracy | **82.29%** |
| Test Macro F1 | **70.90%** |
| Test Weighted F1 | **82.73%** |
| Production artifact path | models/resnet50/best_resnet50.pth in S3 |

This V3 record supersedes the older benchmark numbers previously documented in this README.

---

# 📦 Model Artifact

Training produces:

```text
models/
└── best_resnet50.pth
```

The checkpoint stores more than the raw weights. The serving layer uses these fields to reconstruct the model and class mapping:

```text
artifact_version
model_name
state_dict
class_names
num_classes
image_size
mean
std
epoch
val_accuracy
val_macro_f1
phase
```

This makes the model artifact **self-describing** and allows the serving layer to reconstruct the correct architecture, preprocessing assumptions, and class mapping.

The production API creates the architecture with:

```python
NepaliDressClassifier(
    num_classes=num_classes,
    pretrained=False,
)
```

The `pretrained=False` setting is important in deployment: the API loads the trained checkpoint instead of downloading ImageNet weights again.

---

# ☁️ AWS S3 Model Storage

The model artifact is separated from the application container.

```text
Amazon S3
    │
    │ startup download
    ▼
FastAPI container
    │
    ▼
ResNet50
```

Default configuration:

```text
S3_MODEL_KEY=models/resnet50/best_resnet50.pth
AWS_REGION=ap-southeast-2
```

The bucket is configurable with:

```bash
S3_BUCKET=your-bucket
S3_MODEL_KEY=models/resnet50/best_resnet50.pth
AWS_REGION=ap-southeast-2
```

The application uses boto3's AWS credential chain. On EC2, the recommended approach is an **IAM instance role** with least-privilege access to the required S3 object.

---

# 🚀 FastAPI Inference API

The API lives in:

```text
api/
├── main.py
├── predictor.py
└── schemas.py
```

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information and available routes |
| GET | `/health` | Service and model health |
| GET | `/model-info` | Architecture, class count, classes, device, checkpoint metadata |
| POST | `/predict` | Classify an uploaded image |

Interactive documentation:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

---

# 🔍 Prediction Pipeline

```text
JPEG / PNG / WEBP
        │
        ▼
Content-type validation
        │
        ▼
5 MB size validation
        │
        ▼
PIL image validation
        │
        ▼
RGB conversion
        │
        ▼
224 × 224 resize
        │
        ▼
ImageNet normalization
        │
        ▼
ResNet50 inference
        │
        ▼
Softmax probabilities
        │
        ▼
Top-3 ranking
        │
        ▼
0.60 confidence threshold
        │
        ├── confidence ≥ 0.60 → success
        │
        └── confidence < 0.60 → uncertain
```

The API accepts:

- JPEG
- PNG
- WEBP

The application enforces a **5 MB maximum image size** and also validates that the uploaded bytes are a real readable image rather than relying only on the MIME type.

---

# 📋 Example Response

```json
{
  "filename": "nepali-dress.jpg",
  "status": "success",
  "top_prediction": "example_class",
  "confidence": 94.21,
  "top_k": [
    {
      "class_name": "example_class",
      "confidence": 94.21
    },
    {
      "class_name": "another_class",
      "confidence": 3.71
    },
    {
      "class_name": "third_class",
      "confidence": 1.02
    }
  ],
  "is_out_of_distribution": false
}
```

### Uncertain prediction

When the highest softmax confidence is below `0.60`:

```json
{
  "status": "uncertain",
  "is_out_of_distribution": true
}
```

> **Important:** this is a confidence-based uncertainty flag, not a formally calibrated or guaranteed OOD detector. Softmax confidence should not be interpreted as a statistically calibrated probability of correctness.

---

# 🐳 Docker

The inference service is containerized for repeatable deployment.

Build:

```bash
docker build -t nepali-dress-api .
```

Run:

```bash
docker run -d \
  --name nepali_dress_api \
  -p 9000:8000 \
  -e AWS_REGION=ap-southeast-2 \
  -e S3_BUCKET=your-bucket \
  -e S3_MODEL_KEY=models/resnet50/best_resnet50.pth \
  nepali-dress-api
```

Or:

```bash
docker compose up --build
```

The local container mapping used by the deployment configuration is:

```text
Host     : 9000
Container: 8000
```

---

# ☁️ Amazon EC2 Deployment

The production deployment runs the Dockerized FastAPI service on **Amazon EC2**.

High-level architecture:

```text
                    GitHub
                      │
                      │ push
                      ▼
              GitHub Actions
                      │
                      │ SSH
                      ▼
                 Amazon EC2
                      │
                      ▼
               Docker Container
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
          FastAPI           Amazon S3
             │              Model Artifact
             ▼
          ResNet50
```

The deployment workflow:

1. Checks out the latest `main` branch
2. Connects to EC2
3. Removes the previous application container
4. Clones the current repository
5. Verifies required files
6. Verifies deployment model configuration
7. Builds the Docker image
8. Starts the container
9. Waits for the API to become ready
10. Requires `model_loaded=true`
11. Queries `/model-info`
12. Performs a final health verification
13. Cleans unused Docker images
14. Fails the deployment if health verification fails

Workflow:

```text
.github/workflows/deploy.yml
```

---

# 🔄 CI/CD

A push to `main` can trigger the EC2 deployment workflow.

The deployment is deliberately **health-gated**.

A successful Docker build alone is not considered a successful deployment.

The workflow verifies:

```text
Docker build
     ↓
Container running
     ↓
/health responds
     ↓
model_loaded = true
     ↓
/model-info responds
     ↓
Final health check
     ↓
DEPLOYMENT SUCCESSFUL
```

If the model cannot be downloaded from S3 or cannot be loaded into ResNet50, the deployment fails instead of silently publishing an unhealthy API.

---

# 🔐 Security & Production Practices

The repository follows practical production engineering principles:

- AWS access keys are not embedded in Python source code
- EC2 can use an IAM instance role
- S3 model storage is separated from application code
- Docker runs the API as a non-root user
- Uploaded images are size-limited
- MIME types are validated
- Image bytes are actually decoded before inference
- A model-loading failure prevents the application from completing startup
- Runtime configuration is supplied through environment variables
- CI/CD secrets are expected to remain in GitHub Secrets
- S3 access should follow least privilege

### Recommended IAM design

The EC2 role should have only the S3 permissions required to read the model artifact.

Avoid attaching broad administrator permissions to the application instance.

---

# 🧪 Local Setup

## 1. Clone the repository

```bash
git clone https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition.git
cd nepali-cultural-dress-recognition
```

## 2. Create a virtual environment

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

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For the CPU-oriented Docker runtime:

```bash
pip install -r requirements-docker.txt
```

## 4. Validate the dataset

```bash
python -m src.dataset_prep
```

## 5. Train

```bash
python -m src.train
```

## 6. Evaluate

```bash
python -m src.evaluate
```

## 7. Start the API

Configure AWS access and model location:

### Windows PowerShell

```powershell
$env:AWS_REGION="ap-southeast-2"
$env:S3_BUCKET="your-bucket"
$env:S3_MODEL_KEY="models/resnet50/best_resnet50.pth"

uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# 📁 Repository Structure

```text
nepali-cultural-dress-recognition/
│
├── .github/
│   └── workflows/
│       └── deploy.yml              # EC2 deployment automation
│
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application
│   ├── predictor.py                # S3 + model + inference service
│   └── schemas.py                  # Pydantic response schemas
│
├── src/
│   ├── __init__.py
│   ├── dataset_prep.py             # Dataset validation and class checks
│   ├── evaluate.py                 # Test evaluation + artifacts
│   ├── model.py                    # ResNet50 architecture
│   ├── train.py                    # Two-phase training
│   └── utils.py                    # Reserved utility module
│
├── Dockerfile                      # Production container
├── docker-compose.yml              # Local container orchestration
├── requirements.txt                # Development/training dependencies
├── requirements-docker.txt         # CPU deployment dependencies
├── .dockerignore
├── .gitignore
└── README.md
```

---

# 🧠 Engineering Decisions

### Why ResNet50?

ResNet50 provides a strong ImageNet pretrained representation and is a practical foundation for transfer learning on a relatively specialized visual dataset.

### Why two-stage fine-tuning?

Training the head first stabilizes the new classification layer. Unfreezing `layer4` afterward allows domain adaptation without aggressively changing the entire pretrained backbone.

### Why Macro F1?

Accuracy can hide poor performance on smaller classes. Macro F1 gives each class equal importance and is therefore useful for model selection on imbalanced multi-class data.

### Why inverse-square-root class weighting?

Direct inverse-frequency weighting can become excessively aggressive for rare classes. Inverse-square-root weighting provides a softer correction.

### Why S3?

The model artifact is independent from the application image. A model can therefore be replaced without embedding a large checkpoint directly into the source repository or Docker build context.

### Why Docker?

Docker gives the inference service a reproducible runtime and makes local-to-cloud deployment more consistent.

### Why EC2?

EC2 provides direct control over the runtime, Docker environment, networking, and IAM integration.

### Why health-gated deployment?

A container that starts successfully is not necessarily a working ML service. The deployment therefore verifies that the API is responding **and that the model is actually loaded**.

---

# 📦 Reproducibility

The training pipeline uses:

```text
SEED = 42
```

and records important checkpoint metadata.

For serious experiment reproduction, record:

```text
Dataset version
   +
Git commit
   +
Python/dependency versions
   +
Training configuration
   +
Model checkpoint
```

The checkpoint itself contains the class mapping and preprocessing metadata required by the serving pipeline.

---

# ⚠️ Limitations

This system is a **closed-set image classifier** for the classes represented in the training data. It is not a general-purpose cultural understanding model.

Current limitations include:

- performance depends on dataset quality and diversity
- the test split does not currently contain examples for every model class
- softmax confidence is not calibrated probability
- the uncertainty mechanism is threshold-based rather than a formal OOD model
- predictions outside the learned class distribution may still be assigned to a known class
- CPU inference is slower than GPU inference
- AWS/S3 access is required during production startup
- the model is currently optimized for a fixed 24-class label space and the current test split does not cover all 24 classes

These limitations are documented deliberately so that benchmark numbers are not overstated.

---

# 🚧 Future Roadmap

Potential improvements:

- [ ] Add complete test coverage for all 24 classes
- [ ] Calibrate confidence scores
- [ ] Add a dedicated OOD detection method
- [ ] Add automated API integration tests
- [ ] Add latency and throughput benchmarks
- [ ] Add model versioning
- [ ] Add experiment tracking
- [ ] Add dataset versioning
- [ ] Add model monitoring
- [ ] Quantize the model for faster CPU inference
- [ ] Add GPU inference where appropriate
- [ ] Add HTTPS behind a production reverse proxy
- [ ] Add infrastructure-as-code for AWS
- [ ] Add blue/green or canary deployment
- [ ] Add automatic rollback on failed deployment
- [ ] Add API authentication/rate limiting for public production use

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision + Pillow |
| Architecture | ResNet50 |
| Metrics | scikit-learn |
| API | FastAPI |
| ASGI Server | Uvicorn |
| Validation | Pydantic |
| AWS SDK | boto3 |
| Model Storage | Amazon S3 |
| Compute | Amazon EC2 |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Dataset Format | PyTorch ImageFolder |

---

# 🌟 Why This Project Matters

This project demonstrates the full lifecycle of a machine-learning application:

```text
ML Research
    ↓
Dataset Engineering
    ↓
Transfer Learning
    ↓
Fine-Tuning
    ↓
Evaluation
    ↓
Model Packaging
    ↓
API Engineering
    ↓
Containerization
    ↓
Cloud Storage
    ↓
Cloud Deployment
    ↓
Automated CI/CD
    ↓
Production Health Verification
```

It therefore serves as a portfolio project demonstrating **computer vision + machine learning engineering + backend API development + Docker + AWS + deployment automation** in one system.

---

# 🔗 Project Links

**Repository**

https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition

**API documentation when deployed**

```text
/docs
/redoc
```

---

# 👨‍💻 Author

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

# 📄 License

A formal open-source license has not yet been selected for this repository.

If this project is intended for public reuse, add an appropriate license such as MIT before presenting it as an open-source project.

---

<p align="center">
  <b>🇳🇵 Built to preserve, understand, and digitally recognize Nepal's cultural heritage.</b>
</p>
