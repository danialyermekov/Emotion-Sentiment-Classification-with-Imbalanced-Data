# \# Multilingual Emotion Classification with Data Augmentation

Datasets: https://github.com/sotlampr/universal-joy

This project explores multilingual emotion classification (5 classes: `anger`, `anticipation`, `fear`, `joy`, `sadness`) across multiple languages, with a focus on comparing different data balancing strategies and augmentation techniques. The study includes three main experiments and a comparison with an LLM baseline.

## Overview

The goal of this project is to investigate how different data augmentation and balancing strategies affect model performance in multilingual emotion classification tasks. We compare:

1. **Class-level augmentation** (downsampling large classes, upsampling small classes)
2. **Test set augmentation** for a specific underrepresented class
3. **Proper balanced dataset** with minimum samples per language-class pair
4. **LLM baseline** for benchmarking against state-of-the-art language models

## Experiments

### Experiment 1: Train Set Augmentation by Class

**Objective**: Balance the training data by class size using augmentation techniques.

**Method**:

- Large classes (e.g., `joy`) were downsampled to **8,000 examples**
- Small classes (e.g., `fear`) were upsampled to **4,000 examples** using:
    - Synonym replacement
    - Back-translation (translate to intermediate language and back)

**Languages**: 18 languages (merged 'low resource' and 'small' datasets from https://github.com/sotlampr/universal-joy)

**Results**: See `report.pdf` for detailed classification reports and confusion matrices.

### Experiment 2: Test Set Augmentation for Minority Class

**Objective**: Evaluate model performance when test set for minority class is artificially augmented.

**Method**:

- Same model as Experiment 1
- Test set for `fear` class augmented to **500 examples** using synonym replacement and back-translation
- Other classes remain unchanged

**Note**: This experiment demonstrates how augmented test sets can affect metric interpretation and should be used cautiously.

**Results**: See `report.pdf`

### Experiment 3: Properly Balanced Dataset with Language Reduction

**Objective**: Create a truly balanced dataset with equal representation across language-class pairs.

**Method**:

- Minimum **500 examples** per language per class
- Number of languages reduced from **18 to 11** (focusing on languages with sufficient data)
- Back translation; 

**Languages** (11 total): English, Spanish, French, German, Russian, Arabic, Chinese, Portuguese, Italian, Dutch, Turkish

**Results**: See `report.pdf`

### LLM Baseline Comparison

**Objective**: Compare fine-tuned model performance against Large Language Model (LLM) zero-shot/few-shot/chain-of-thought classification.

**Method**:

- Used Llama 3.1 and Llama 3.3 for emotion classification via prompt engineering
- Test set included one example for each class and language (5 classes and 18 languages)
- Evaluated on precision, recall, F1-score, and accuracy

**Results**: See `report.pdf`

## Dataset

### Format

| Column | Description | Example |
| :-- | :-- | :-- |
| `text` | Input text (sentence or paragraph) | "I am so happy today!" |
| `label` | Emotion class | `joy` |
| `lang` | ISO language code | `en` |

### Classes

- `anger`: Expressions of frustration, annoyance, or rage
- `anticipation`: Expressions of expectation or looking forward to something
- `fear`: Expressions of anxiety, worry, or terror
- `joy`: Expressions of happiness, pleasure, or satisfaction
- `sadness`: Expressions of sorrow, grief, or disappointment


### Data Splits

- **Training set**: 70%
- **Validation set**: 15%
- **Test set**: 15%

Each split maintains language and class distribution proportions.

## Project Structure

.
├── csv/
│ ├── raw/ \# Original unprocessed data
│ ├── augmented/ \# Augmented datasets (Exp 1 \& 2)
│ └── balanced/ \# Balanced dataset (Exp 3)
├── ipynbs/
│ ├── training.ipynb \# Experiment 1,2 notebook
│ └── training1.ipynb \# Experiment 3 notebook
├── requirements.txt
├── README.md
└── report.pdf

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU training, recommended)
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
https://github.com/danialyermekov/Credit-Card-Fraud-Detection-with-MLP-and-Autoencoder.git
cd multilingual-emotion-classification
```
Create a virtual environment:
```bash
venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Groq API for LLMs you can get here: https://groq.com/

## Key Findings
- Class Imbalance Impact: The fear class consistently shows lower performance across all experiments due to limited representation in the original dataset.
- Augmentation Benefits: Experiment 1 shows modest improvements for minority classes through synthetic data generation, but overfitting to augmentation patterns is evident.
- Test Set Augmentation Risk: Experiment 2 demonstrates inflated metrics for the fear class when test set is augmented, highlighting the importance of maintaining realistic test distributions.
- Language Reduction: Focusing on 11 well-represented languages (Experiment 3) provides more stable cross-lingual performance compared to spreading data thin across 18 languages.
- LLM Comparison: LLM baseline shows competitive performance on high-resource languages but struggles with code-switched and informal text.

## Model Architecture

- Base Model
    - Architecture: XLM-RoBERTa (base)
    - Parameters: 270M
    - Pretraining: 100 languages (2.5TB CommonCrawl data)
- Hyperparameters
    - Learning rate: 3e-5
    - Batch size: 16
    - Epochs: 5
    - Warmup ratio: 0.1
    - Weight decay: 0.01
    - Max sequence length: 128
    - Optimizer: AdamW
    - Scheduler: Linear with warmup
- Training Details
    - Hardware: T4
    - Training time: 1-2 hours per experiment

## Evaluation Metrics

- Per-Class Metrics
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
- Aggregate Metrics
    - Accuracy: Overall correct predictions
    - Macro Average: Unweighted mean of per-class metrics (treats all classes equally)
    - Weighted Average: Mean of per-class metrics weighted by support (accounts for class imbalance)

## Additional Analysis

- Confusion matrices for error analysis
- Per-language breakdown of performance

## Future Work

- Advanced Augmentation: Explore GPT-based paraphrasing and multilingual data synthesis
- Balanced Sampling: Implement curriculum learning with dynamic class weighting
- Cross-lingual Transfer: Investigate zero-shot transfer to low-resource languages
- Ensemble Methods: Combine multiple model architectures for improved robustness
- Error Analysis: Deep dive into misclassification patterns across languages
- Real-world Deployment: Build API endpoint for production inference

## References

- XLM-RoBERTa: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (ACL 2020)
- Data Augmentation: Wei \& Zou, "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" (EMNLP 2019)
- Multilingual Sentiment: Mohammad et al., "SemEval-2018 Task 1: Affect in Tweets" (SemEval 2018)
