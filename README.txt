# Metal Nut Defect Detection Using Deep Learning

A computer vision project for detecting defects in metal nuts using a compact deep learning model. The project includes dataset preparation, bounding-box generation from segmentation masks, model training, evaluation, failure-case analysis, ONNX export, and CPU inference benchmarking.

## 1. Project Overview

This project was developed as part of the Computer Vision Intern take-home assignment for InSiSo Technologies.

The objective is to identify defective metal nuts, predict their bounding boxes, evaluate detection performance, and export the trained model to ONNX format for efficient inference.

**Key features**
- Dataset preparation and image validation
- Bounding-box label generation from defect masks
- Custom CNN-based defect classification and localization
- Precision, recall, and mAP@0.5 evaluation
- Failure-case analysis
- ONNX model export
- CPU inference latency benchmarking
- Baseline versus improved model comparison

## 2. Dataset

The project uses the `metal_nut` category from the MVTec Anomaly Detection (MVTec AD) dataset.

The dataset contains normal metal nuts and defective samples, including defect categories such as:
- Bent
- Color
- Flip
- Scratch
- Good (non-defective)

Defect segmentation masks are used to derive bounding boxes around defect regions.

**Preprocessing**
- Images are resized to 128 × 128 pixels.
- Images are converted to RGB format.
- Bounding boxes are generated from the corresponding defect masks.
- Dataset manifests are used to organize training, validation, and testing samples.

**Important evaluation note:** The project uses a custom split of the available dataset, rather than the official MVTec AD benchmark protocol. Therefore, the reported metrics should be interpreted as results on this custom split and not as official benchmark results.

## 3. Project Workflow

1. Load and inspect the dataset.
2. Validate images and remove corrupt files.
3. Generate bounding boxes from defect masks.
4. Create training, validation, and testing manifests.
5. Resize and preprocess images.
6. Train the baseline model.
7. Evaluate the baseline model.
8. Fine-tune the model using an improved training configuration.
9. Compare baseline and improved results.
10. Analyze failure cases.
11. Export the improved model to ONNX.
12. Benchmark CPU inference latency.

## 4. Model Architecture

A compact custom convolutional neural network is used for defect classification and bounding-box regression.

The model has two output branches:

- **Classification branch:** Predicts whether an image contains a defect.
- **Bounding-box regression branch:** Predicts the normalized coordinates of the defect bounding box.

The model contains approximately 276,773 trainable parameters.

### Training configuration

| Parameter | Configuration |
|---|---|
| Input image size | 128 × 128 |
| Image channels | 3 (RGB) |
| Baseline training | 15 epochs |
| Baseline learning rate | 0.001 |
| Improved fine-tuning | 20 epochs |
| Improved learning rate | 0.0003 |
| Optimizer | Adam |
| Classification loss | Binary cross-entropy |
| Bounding-box loss | Smooth L1 |
| Improved box-loss weight | 10 |
| Improved weight decay | 0.0001 |
| Gradient clipping | 5.0 |

## 5. Evaluation Results

The model was evaluated on the custom held-out test split.

| Metric | Baseline | Improved |
|---|---:|---:|
| Precision | 0.1613 | 0.2500 |
| Recall | 0.2632 | 0.3158 |
| mAP@0.5 | 0.1256 | 0.1733 |
| Mean IoU on defective images | 0.3406 | 0.3358 |

### Results interpretation

The improved model increased precision from 0.1613 to 0.2500, recall from 0.2632 to 0.3158, and mAP@0.5 from 0.1256 to 0.1733.

However, mean IoU decreased slightly from 0.3406 to 0.3358. This indicates that localization remains a significant challenge, even though the overall detection metrics improved.

The results show that further work is needed before deploying the model in a production inspection system.

## 6. Improvement Experiment

The baseline model was fine-tuned using a lower learning rate, a higher bounding-box loss weight, weight decay, gradient clipping, and a learning-rate scheduler.

The experiment aimed to improve defect detection and localization.

**Outcome:** Precision, recall, and mAP@0.5 improved. Mean IoU decreased slightly, so the experiment did not improve every metric.

Further experiments could include:
- Transfer learning from a pretrained detector
- Higher-resolution inputs
- More balanced sampling
- Data augmentation
- Improved bounding-box regression
- Anomaly detection trained primarily on normal samples

## 7. Failure-Case Analysis

At least five failure cases were examined to understand the limitations of the model.

Observed issues include:

- **Poor localization:** The model assigns a high defect score but predicts a bounding box that overlaps poorly with the ground-truth box.
- **Classification errors:** Some defective images receive low defect confidence scores.
- **Small or subtle defects:** Fine scratches and minor visual changes may be difficult to localize at 128 × 128 resolution.
- **Appearance variation:** Differences in orientation, color, and defect shape may affect predictions.

The `failure_cases.csv` file contains examples and associated prediction scores and IoU values.

These failure cases demonstrate that high classification confidence does not necessarily mean that the predicted bounding box is accurate.

## 8. ONNX Export and CPU Benchmark

The improved model was exported to ONNX format and tested using ONNX Runtime with the CPU execution provider.

| Parameter | Result |
|---|---|
| Exported model | `metal_nut_detector.onnx` |
| Input shape | 1 × 3 × 128 × 128 |
| Warm-up runs | 10 |
| Timed runs | 100 |
| Average inference latency | Approximately 1.218 ms/image |
| Throughput | Approximately 820.95 images/second |

**Benchmark limitation:** The CPU model name was not recorded. The benchmark used a synthetic input tensor and measures model inference only; image loading and preprocessing are excluded. Therefore, this result should not be treated as end-to-end camera-to-prediction latency or a guarantee of performance on another CPU.

## 9. Repository Structure

```text
metal-nut-defect-detection/
├── README.md
├── improved_metal_nut_detector.pth
├── metal_nut_detector.onnx
├── bounding_boxes.csv
├── train_manifest.csv
├── val_manifest.csv
├── test_manifest.csv
├── metrics.json
└── failure_cases.csv
```

Add your training and evaluation notebook or Python scripts to the repository if they are available. Update this structure to match the files actually uploaded to GitHub.

## 10. Reproducibility

The dataset is not included in this repository. Obtain the MVTec AD dataset separately and place the `metal_nut` directory in the expected location.

The original development environment used Google Colab. The dataset and project files were stored under `/content/`.

To reproduce the experiment:

1. Set up Python and install the required dependencies.
2. Download and extract the MVTec AD dataset.
3. Run the data-preparation and bounding-box generation code.
4. Generate the train, validation, and test manifests.
5. Run the baseline training procedure.
6. Run the improved fine-tuning procedure.
7. Evaluate the model on the held-out custom test split.
8. Export the model to ONNX and run the CPU benchmark.

Exact commands should be added after the training and evaluation code has been organized into runnable notebook cells or Python scripts.

## 11. Limitations and Future Work

Current limitations include:
- Limited training data for some defect categories
- Low detection precision and recall
- Weak bounding-box localization
- Custom rather than official benchmark splitting
- CPU benchmark performed without image-loading and preprocessing time
- No validation yet on a real production inspection line

Future improvements could include a pretrained object-detection model, more suitable data augmentation, improved localization loss, threshold calibration, higher-resolution inputs, and evaluation on the official benchmark split.

## 12. AI Assistance Disclosure

AI assistance was used during development to help with code, debugging, documentation, and interpretation of results. The implementation, experimental results, limitations, and final submission should be reviewed and understood by the author.

## 13. Acknowledgements

Dataset: MVTec Anomaly Detection (MVTec AD), `metal_nut` category.

Project prepared for the Computer Vision Intern take-home assignment at InSiSo Technologies.
