# Thesis Summary (Draft)

> **This file is a draft scaffold.** It's built from what's verifiable in the training notebooks. Share your actual thesis report (PDF/DOCX) and I can rewrite this with your real abstract, literature review framing, and final reported numbers — replacing the placeholders below.

## Title
Comparative CNN-Based Plant Disease Classification using Transfer Learning

## Institution
JNTUH University College of Engineering, Science and Technology, Hyderabad — M.Tech (Integrated Dual Program), Computer Science

## Supervisor
Dr. Athota Kavitha

## Problem Statement
*[Replace with your thesis's framing — e.g., crop loss due to late disease diagnosis, need for an automated, low-cost classification system for farmers.]*

## Methodology
1. **Dataset**: PlantVillage (unaugmented), 39 classes, ~54,306 images, 14 plant species.
2. **Preprocessing**: `ImageDataGenerator` with `rescale=1./255`, 80/20 train-validation split.
3. **Models**: Transfer learning on 4 ImageNet-pretrained backbones (ResNet50, VGG16, MobileNetV2, EfficientNetB0), each with a frozen convolutional base and a custom dense classification head.
4. **Training**: Adam optimizer (`lr=1e-4`), `categorical_crossentropy` loss, `EarlyStopping` + `ReduceLROnPlateau` + `ModelCheckpoint` callbacks. Training executed on Kaggle's free GPU tier; evaluation and real-image testing executed on Google Colab.
5. **Evaluation**: Validation accuracy/loss, per-class precision/recall/F1 (`classification_report`), confusion matrices, and a held-out real-world image test.

## Key Findings
- See the [Results table in the root README](../README.md#-results) for the full comparison.
- **MobileNetV2** offered the best accuracy-to-model-size tradeoff (~97% val. accuracy at ~12.5MB), making it the most deployment-practical of the four.
- **VGG16** achieved comparably high accuracy but at a far larger model footprint.
- **EfficientNetB0** underperformed in this run — flagged for further investigation (see README note on preprocessing).

## Conclusion
*[Replace with your thesis's actual conclusion/contribution statement.]*

## Future Work
*[Replace with your thesis's stated future work section, or see the README's Future Work section for engineering-focused next steps.]*
