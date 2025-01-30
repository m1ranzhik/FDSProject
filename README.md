# ADHD Classification using Random Forest, LSTM, and CNN

This project implements machine learning and deep learning models (Random Forest, LSTM, and CNN) to classify ADHD diagnosis using functional MRI (fMRI) and task performance data. The project preprocesses fMRI data, extracts features, and trains models to predict ADHD status.

---

## Features and Implementation

### Data Loading and Preprocessing
[Dataset link](https://openneuro.org/datasets/ds003500/versions/1.2.0)
- **Participants Information**: Reads participant metadata from `participants.tsv`.
- **fMRI Files**: Extracts BOLD activity and masks regions for feature extraction.
- **Event Files**: Analyzes task-based event data, such as `go` and `no-go` counts and response times.

### Feature Engineering
- Combines fMRI mean activation, task performance features (`go`, `no-go`, response time), and other metrics.
- Standardizes features for model compatibility.

### Models
1. **Random Forest**:
   - Uses extracted features to classify ADHD.
   - Outputs feature importance and performance metrics.

2. **Convolutional Neural Network (CNN)**:
   - Processes 3D fMRI data directly for classification.
   - Includes Conv3D, MaxPooling3D, Flatten, Dense, and Dropout layers.

3. **LSTM**:
   - Sequentially processes time-series data from fMRI for ADHD prediction.
   - Optimized using the Adam optimizer.

## Dependencies
- Python 3.10+
- Libraries: `numpy`, `pandas`, `nibabel`, `nilearn`, `sklearn`, `tensorflow`, `matplotlib`


