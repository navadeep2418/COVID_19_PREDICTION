# COVID-19 Daily Case Prediction Using LSTM Deep Learning

> Deep Learning Course Project | Apollo University | CSE Department | AY 2025-26

A stacked LSTM neural network trained on real Johns Hopkins CSSE COVID-19 data (India, 2020–2022) to predict daily confirmed cases across three pandemic waves.

## 🌐 Live Demo
[View Showcase →](https://covid19-lstm-deeplearning.vercel.app)

## 📊 Model Results
| Metric | Value |
|--------|-------|
| R² Score | 0.87 |
| RMSE | ~14,823 cases/day |
| MAE | ~9,641 cases/day |
| MAPE | ~18.4% |

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python covid19_prediction.py
```

## 📁 Project Structure
```
├── index.html                  # Vercel showcase page
├── covid19_prediction.py       # Main deep learning script
├── requirements.txt            # Python dependencies
└── README.md
```

## 🧠 Model Architecture
- LSTM Layer 1 → 128 units + Dropout(0.2)
- LSTM Layer 2 → 64 units + Dropout(0.2)
- Dense → 32 units (ReLU)
- Dense → 1 unit (output)

## 📦 Dependencies
- TensorFlow 2.x / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## 📄 Dataset
[Johns Hopkins CSSE COVID-19 Data](https://github.com/CSSEGISandData/COVID-19)

## 👨‍🏫 Submitted To
Dr. K Sudheer | Apollo University
