# ECG Anomaly Detection with Deep Learning Autoencoders

A deep learning mini project for detecting anomalies in ECG (Electrocardiogram) signals using GRU and LSTM-based autoencoders. This project features a Streamlit web app for interactive analysis, pretrained models, and visualizations of model performance.

## Features
- GRU and LSTM autoencoder architectures implemented in PyTorch
- Anomaly detection on ECG time-series data
- Interactive Streamlit web app for:
  - Uploading and analyzing ECG data
  - Visualizing reconstruction and errors
  - Batch and individual sample analysis
  - Downloadable results
- Pretrained model weights and convergence plots

## Project Structure
```
dl-mini-projet-main/
├── deployment/         # Streamlit app and deployment files
├── docs/              # Project documentation
├── models/            # Pretrained model weights (.pth)
├── plots/             # Training and convergence plots
├── ecg-detection.ipynb# Main Jupyter notebook (model training & analysis)
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Streamlit
- pandas, numpy, matplotlib

Install dependencies:
```bash
pip install torch streamlit pandas numpy matplotlib
```

### Running the Streamlit App
1. Navigate to the `deployment` directory:
   ```bash
   cd dl-mini-projet-main/deployment
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Upload your ECG CSV file (140 features + 1 label column) via the sidebar.

### Notebook Usage
- Open `ecg-detection.ipynb` for model training, evaluation, and further analysis.

## Dataset
- The project expects ECG data in CSV format, with 140 feature columns (signal values) and 1 label column (1=Normal, 0=Anomaly).
- Example/test datasets are provided in the `deployment/` directory.

## Results & Visualizations
- Training and convergence plots are available in the `plots/` directory.
- The Streamlit app provides:
  - Reconstruction error visualization
  - Anomaly/normal prediction
  - Batch accuracy metrics

## Credits
- Developed as a Deep Learning mini project.
