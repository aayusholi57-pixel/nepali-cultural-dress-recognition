# 🇳🇵 Nepali Cultural Dress Recognition

> **Production-oriented computer vision system for recognizing Nepali cultural dresses and ornaments using a fine-tuned ResNet50 model, FastAPI, Docker, Amazon S3, Amazon EC2, and GitHub Actions.**

[![CI/CD](https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition/actions/workflows/deploy.yml/badge.svg)](https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition/actions/workflows/deploy.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/Cloud-AWS-232F3E?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)

---

## Overview

**Nepali Cultural Dress Recognition** is an end-to-end image classification platform designed to identify Nepali cultural clothing and ornaments from uploaded images.

The project combines a transfer-learning training pipeline with a production API and cloud-based model delivery:

**Image → preprocessing → fine-tuned ResNet50 → top-k predictions → confidence/uncertainty decision → FastAPI response**

The trained checkpoint is stored outside the application container in **Amazon S3**. At application startup, the API downloads the model artifact from S3, reconstructs the ResNet50 architecture, loads the trained state dictionary, and exposes inference through a documented FastAPI service.

The deployment stack is containerized with Docker and automated through GitHub Actions to an Amazon EC2 instance.

---

## Key Capabilities

- 🧠 **ResNet50 transfer learning** for visual classification
- 🎯 **Two-stage fine-tuning**
  - Phase 1: classification head training
  - Phase 2: ResNet50 layer4 + classification head fine-tuning
- ⚖️ **Class-imbalance handling** using inverse-square-root class weighting
- 🖼️ **Image augmentation** for improved generalization
- 📊 **Accuracy, Macro F1, and Weighted F1** tracking
- 🔎 **Top-k predictions** with confidence scores
- 🚦 **Confidence-based uncertainty / OOD-style flagging**
- ☁️ **Amazon S3 model artifact storage**
- 🚀 **Amazon EC2 deployment**
- 🐳 **Dockerized inference service**
- 🔄 **GitHub Actions CI/CD**
- ❤️ **Health and model-information endpoints**
- 🔐 **AWS credential-chain support** suitable for EC2 IAM roles
- 🛡️ **Non-root Docker runtime**
- 📦 **Reproducible checkpoint metadata**
- 🧪 **Dataset validation and class-consistency checks**

---

## System Architecture

~~~text
                         ┌──────────────────────┐
                         │   Client / Browser    │
                         └──────────┬───────────┘
                                    │
                                    │ image upload
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI API      │
                         │   /predict /health    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   PredictorService    │
                         │ preprocessing +       │
                         │ inference + top-k     │
                         └──────────┬───────────┘
                                    │
                                    │ trained weights
                                    ▼
                         ┌──────────────────────┐
                         │     ResNet50          │
                         │  fine-tuned model     │
                         └──────────────────────┘
                                    ▲
                                    │
                          startup download
                                    │
                         ┌──────────────────────┐
                         │     Amazon S3         │
                         │ best_resnet50.pth     │
                         └──────────────────────┘

        GitHub ──► GitHub Actions ──► Docker build ──► Amazon EC2
~~~

---

## Machine Learning Pipeline

### 1. Dataset validation

The dataset follows an ImageFolder-compatible structure:

~~~text
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
~~~

The dataset validation script checks:

- required train/validation/test splits
- class-folder consistency
- supported image extensions
- image counts per class
- empty class directories
- train/validation/test class mismatches
- reproducible class-index mapping

Run:

~~~bash
python -m src.dataset_prep
~~~

---

### 2. Image preprocessing

Training images use augmentation including:

- Random resized crop
- Horizontal flip
- Small random rotation
- Color jitter
- Tensor conversion
- ImageNet normalization

Validation and inference use deterministic resizing and ImageNet normalization.

Target image size:

~~~text
224 × 224
~~~

---

### 3. Model architecture

The core classifier is **ResNet50** initialized with ImageNet pretrained weights during training.

The original classification layer is replaced with:

~~~text
Dropout
   ↓
Linear
   ↓
ReLU
   ↓
BatchNorm
   ↓
Dropout
   ↓
Linear → N classes
~~~

This allows the ImageNet representation to be adapted to the Nepali cultural dress domain.

---

## Fine-Tuning Strategy

The training pipeline uses two controlled optimization phases.

### Phase 1 — Classification head

The ResNet50 backbone is frozen.

Only the custom classification head is trained.

~~~text
Backbone      → Frozen
Classifier    → Trainable
Learning rate → 1e-3
~~~

This allows the new classifier to first learn the target label space without aggressively changing the pretrained visual representation.

### Phase 2 — Domain adaptation

The final ResNet50 feature block, layer4, is unfrozen together with the classification head.

~~~text
Early backbone → Frozen
layer4         → Trainable
Classifier     → Trainable
~~~

The optimizer uses separate learning rates:

~~~text
layer4 → 1e-5
head   → 1e-4
~~~

This is genuine **fine-tuning of pretrained ResNet50**, rather than training a new CNN from scratch.

Run training with:

~~~bash
python -m src.train
~~~

The best checkpoint is selected using **validation Macro F1**.

---

## Training Configuration

| Component | Configuration |
|---|---|
| Architecture | ResNet50 |
| Initialization | ImageNet pretrained weights |
| Input size | 224 × 224 |
| Batch size | 8 |
| Phase 1 epochs | 3 |
| Phase 2 epochs | 3 |
| Head learning rate | 1e-3 |
| Layer4 learning rate | 1e-5 |
| Fine-tuned head learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Loss | Weighted Cross Entropy |
| Selection metric | Validation Macro F1 |
| Random seed | 42 |
| Runtime training target | CPU-compatible |

> Training configuration is intentionally kept explicit in src/train.py so experiments remain reproducible and auditable.

---

## Evaluation

The evaluation pipeline loads the saved checkpoint and evaluates the model against the test split.

It supports:

- classification report
- confusion matrix
- per-class analysis
- accuracy-oriented evaluation
- Macro F1
- Weighted F1
- evaluation artifact generation

Run:

~~~bash
python -m src.evaluate
~~~

Evaluation outputs are written under:

~~~text
evaluation/
~~~

The README intentionally does **not** hard-code a benchmark score. The reported metric should come directly from the latest reproducible evaluation run rather than being manually copied into documentation.

---

## Model Artifact

The training pipeline produces:

~~~text
models/
└── best_resnet50.pth
~~~

The checkpoint is a self-contained artifact containing:

- model architecture identifier
- artifact version
- trained state_dict
- class names
- number of classes
- image size
- normalization parameters
- training epoch
- validation accuracy
- validation Macro F1
- training phase

This metadata allows the inference service to reconstruct the model consistently with the training configuration.

---

## Amazon S3 Model Management

The production API does not require the model checkpoint to be baked into the Docker image.

Instead:

~~~text
Amazon S3
   │
   │ download at application startup
   ▼
FastAPI container
   │
   ▼
ResNet50 inference
~~~

Default model configuration:

~~~text
S3_MODEL_KEY=models/resnet50/best_resnet50.pth
~~~

The bucket and object key can be configured through environment variables:

~~~bash
S3_BUCKET=your-bucket
S3_MODEL_KEY=models/resnet50/best_resnet50.pth
AWS_REGION=ap-southeast-2
~~~

For EC2, the application supports the standard AWS credential chain and can use an **IAM role attached to the instance**, avoiding hard-coded AWS access keys inside the application.

---

## FastAPI Service

The API is implemented in:

~~~text
api/
├── main.py
├── predictor.py
└── schemas.py
~~~

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | API information |
| GET | /health | Service/model health |
| GET | /model-info | Model metadata and supported classes |
| POST | /predict | Image classification |

Interactive documentation is automatically provided by FastAPI:

~~~text
/docs
/redoc
~~~

### Example prediction flow

~~~text
JPEG / PNG / WEBP
        │
        ▼
File validation
        │
        ▼
RGB conversion
        │
        ▼
224×224 preprocessing
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
Top-3 predictions
        │
        ▼
Confidence threshold
        │
        ▼
JSON response
~~~

The service limits uploaded images to **5 MB** and rejects unsupported image formats.

---

## Example API Response

A successful response follows this general structure:

~~~json
{
  "filename": "dress.jpg",
  "status": "success",
  "top_prediction": "example_class",
  "confidence": 94.21,
  "top_k": [
    {
      "class_name": "example_class",
      "confidence": 94.21
    }
  ],
  "is_out_of_distribution": false
}
~~~

When the highest confidence is below the configured threshold, the API marks the result as uncertain instead of presenting the prediction as highly confident.

---

## Local Development

### 1. Clone

~~~bash
git clone https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition.git
cd nepali-cultural-dress-recognition
~~~

### 2. Create a virtual environment

Windows:

~~~powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
~~~

Linux/macOS:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 4. Validate the dataset

~~~bash
python -m src.dataset_prep
~~~

### 5. Train

~~~bash
python -m src.train
~~~

### 6. Evaluate

~~~bash
python -m src.evaluate
~~~

### 7. Start the API

The API expects access to the trained model through S3 in production.

For local inference, configure AWS credentials and the model location:

~~~powershell
$env:AWS_REGION="ap-southeast-2"
$env:S3_BUCKET="your-bucket"
$env:S3_MODEL_KEY="models/resnet50/best_resnet50.pth"

uvicorn api.main:app --reload
~~~

Open:

~~~text
http://127.0.0.1:8000/docs
~~~

---

## Docker

The repository includes a production-oriented Dockerfile and Compose configuration.

Build:

~~~bash
docker build -t nepali-dress-api .
~~~

Run:

~~~bash
docker run -d \
  --name nepali_dress_api \
  -p 9000:8000 \
  -e AWS_REGION=ap-southeast-2 \
  -e S3_BUCKET=your-bucket \
  -e S3_MODEL_KEY=models/resnet50/best_resnet50.pth \
  nepali-dress-api
~~~

Or use Compose:

~~~bash
docker compose up --build
~~~

The service is exposed locally on:

~~~text
http://127.0.0.1:9000
~~~

---

## AWS Deployment Architecture

The production deployment uses:

~~~text
                    GitHub
                      │
                      │ push to main
                      ▼
              GitHub Actions
                      │
                      │ SSH deployment
                      ▼
                 Amazon EC2
                      │
                Docker container
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     FastAPI API             AWS S3
          │                 model artifact
          │                       │
          └────── ResNet50 ◄─────┘
~~~

### Deployment characteristics

- Docker image built on the EC2 host
- Application container runs as a non-root user
- Model artifact remains in S3
- AWS region defaults to ap-southeast-2
- Container restart policy is enabled
- Docker health checks are configured
- CI/CD verifies the container and /health endpoint
- Deployment fails if the model does not load successfully

The deployment workflow is defined in:

~~~text
.github/workflows/deploy.yml
~~~

---

## CI/CD

The GitHub Actions workflow performs an automated deployment to EC2.

The deployment pipeline includes:

1. Checkout the latest repository state
2. Connect to EC2 through SSH
3. Remove the previous application container
4. Clone the latest main branch
5. Verify required project files
6. Validate model configuration
7. Build the Docker image
8. Start the container
9. Check container health
10. Verify model_loaded=true
11. Query model information
12. Perform final health verification
13. Clean unused Docker images
14. Report deployment success or failure

This makes deployment failures visible at the CI/CD layer instead of silently leaving an unhealthy container running.

---

## Security Design

The project follows several practical production-security principles:

- AWS credentials are not hard-coded into Python source
- EC2 can use an IAM instance role
- Model artifacts are separated from application code
- Docker runs the API as a non-root user
- Uploaded files are size-limited
- Uploaded file MIME types are validated
- Invalid image data is rejected
- Model loading failures prevent false-positive health status
- Secrets are expected to be supplied through runtime configuration or GitHub Secrets

### Recommended AWS policy

For production, the EC2 role should receive the minimum S3 permissions required to read the model artifact rather than broad administrator access.

---

## Project Structure

~~~text
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
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .dockerignore
├── .gitignore
└── README.md
~~~

Large datasets and trained model artifacts are intentionally managed outside the normal application source tree when appropriate.

---

## Engineering Decisions

### Why ResNet50?

ResNet50 provides a strong pretrained visual representation while remaining practical for transfer learning and CPU-based inference.

### Why two-phase fine-tuning?

Training the head first stabilizes the new classifier. Fine-tuning the final backbone block afterward allows the representation to adapt to the target cultural-dress domain while reducing the risk of destroying useful pretrained features.

### Why Macro F1?

A multi-class cultural-dress dataset may not contain perfectly balanced classes. Macro F1 gives every class equal importance and therefore provides a more informative model-selection signal than accuracy alone.

### Why S3 for the model?

Separating model artifacts from the application image makes model replacement easier and avoids rebuilding the container every time the model artifact changes.

### Why Docker?

Docker provides a repeatable runtime containing the API, Python dependencies, and inference environment.

### Why EC2?

EC2 provides direct control over the container runtime, networking, instance configuration, and AWS IAM integration.

### Why GitHub Actions?

Automated deployment reduces manual server operations and ensures the deployed application passes explicit health checks before the workflow reports success.

---

## Reproducibility

The training pipeline records important experiment metadata and uses a fixed random seed:

~~~text
SEED = 42
~~~

The checkpoint stores:

- class mapping
- architecture identifier
- image preprocessing configuration
- validation metrics
- training phase
- epoch information

For a reproducible experiment, keep the dataset version, source commit, Python environment, and model artifact together as part of the experiment record.

---

## Limitations

This is a production-oriented classification system, but it is not a general-purpose cultural understanding model.

Current limitations include:

- predictions depend on the quality and diversity of the training dataset
- confidence is not a calibrated probability of correctness
- the confidence threshold is heuristic
- the system is optimized for the classes represented in the training dataset
- CPU inference can be slower than GPU inference
- the API does not identify arbitrary clothing outside its trained label space
- S3 availability and permissions are required during application startup

---

## Future Improvements

Potential next-stage improvements include:

- calibrated confidence scores
- stronger OOD detection
- model quantization for faster CPU inference
- GPU inference for higher throughput
- automated model versioning
- experiment tracking
- dataset versioning
- model monitoring
- latency and throughput benchmarks
- automated integration tests against the deployed API
- HTTPS behind a production reverse proxy
- infrastructure-as-code for AWS resources
- canary or blue/green deployments
- automated rollback on failed health checks

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Vision | Torchvision |
| Model | ResNet50 |
| ML Metrics | scikit-learn |
| Image Processing | Pillow |
| API | FastAPI |
| API Server | Uvicorn |
| Containerization | Docker |
| Object Storage | Amazon S3 |
| Compute | Amazon EC2 |
| AWS SDK | boto3 |
| Automation | GitHub Actions |
| Dataset Format | ImageFolder |

---

## Repository

**GitHub:**  
https://github.com/aayusholi57-pixel/nepali-cultural-dress-recognition

---

## Author

**Aayush Oli**

AI/ML Engineer in training focused on computer vision, machine learning systems, APIs, cloud deployment, and production-oriented AI engineering.

---

## License

Add the project's chosen open-source license before distributing the repository publicly.
