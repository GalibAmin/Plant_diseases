# Results

Drop exported plots/figures here so they can be embedded in the main `README.md`. Each notebook already generates these — just export and save them with these (or similar) filenames:

| Suggested filename | Source notebook | Cell content |
|---|---|---|
| `class_distribution.png` | `02_eda_and_model_comparison.ipynb` | Bar chart of images per class |
| `vgg16_accuracy_curve.png` | `01_kaggle_training.ipynb` | Train vs. val accuracy over epochs |
| `vgg16_confusion_matrix.png` | `01_kaggle_training.ipynb` | Seaborn heatmap of confusion matrix |
| `model_comparison_bar.png` | `01_kaggle_training.ipynb` | Accuracy/loss bar comparison across models |
| `real_image_prediction.png` | `03_real_image_inference_test.ipynb` | Confidence-by-model bar chart on the real test image |

Once added, reference them in the root README like:
```markdown
![VGG16 Confusion Matrix](results/vgg16_confusion_matrix.png)
```

To export a plot from a running notebook instead of just `plt.show()`:
```python
plt.savefig("results/vgg16_confusion_matrix.png", dpi=150, bbox_inches="tight")
```
