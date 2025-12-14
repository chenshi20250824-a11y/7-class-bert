# 7-class-bert
This project is a **real-time financial news sentiment analysis system** based on the **FinBERT** pre-trained model. It automatically analyzes the impact of financial news on stock prices and predicts stock price movement trends.
# 📰 FinBERT Financial News Sentiment Analysis and Stock Price Prediction System


Real-time Financial News Sentiment Analysis and Stock Price Prediction Based on FinBERT


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![FinBERT](https://img.shields.io/badge/Model-FinBERT-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation & Usage](#installation--usage)
- [User Guide](#user-guide)
- [Project Structure](#project-structure)
- [Demo](#demo)
- [Performance Metrics](#performance-metrics)
- [FAQ](#faq)
- [Limitations and Future Improvements](#️-limitations-and-future-improvements)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Project Overview

This project is a **real-time financial news sentiment analysis system** based on the **FinBERT** pre-trained model. It automatically analyzes the impact of financial news on stock prices and predicts stock price movement trends.And based on the fact that FINBERT can only generate positive, negative, or neutral outputs, we match our news with stock prices, allowing it to roughly predict stock price trends and categorize them into seven types, providing better investment advice.

### Core Features

- 📊 **7-Class Prediction**: Fine-grained classification into 7 categories
- 🌐 **Web Interface**: Beautiful and user-friendly web UI
- ⚡ **Real-time Prediction**: Instant analysis results
- 💡 **Investment Advice**: Provides actionable insights based on predictions

---

## ✨ Features

### 🔢 7-Class Classification System

| Category | Chinese Name | Price Change | Description |
|----------|--------------|--------------|-------------|
| **crash** | 大跌 | < -5% | Major negative news |
| **drop** | 中跌 | -5% ~ -2% | Significant bearish news |
| **small_drop** | 小跌 | -2% ~ -1% | Minor bearish news |
| **stable** | 震荡 | -1% ~ 1% | Neutral news |
| **small_rise** | 小涨 | 1% ~ 2% | Minor bullish news |
| **rise** | 中涨 | 2% ~ 5% | Significant bullish news |
| **surge** | 大涨 | > 5% | Major positive news |

### 🎨 Interface Features

- ✅ Modern purple gradient design
- ✅ Real-time prediction without page refresh
- ✅ Probability visualization with progress bars
- ✅ Example news for quick testing

---

## 🛠️ Tech Stack

### Deep Learning Framework
- **PyTorch** 2.0+
- **Transformers** 4.30+
- **FinBERT** (ProsusAI/finbert)

### Web Framework
- **Flask** 2.3+
- **HTML5 / CSS3 / JavaScript**

### Data Processing
- **Pandas** 1.5+
- **NumPy** 1.24+
- **yfinance** 0.2+

### Development Tools
- **Jupyter Notebook**
- **VS Code**
- **Anaconda**

---

## 📊 Dataset

### Data Sources

- **News Data**: 
  - AlphaVantage API
  - FinViz
  - MarketWatch
- **Stock Price Data**:  Yahoo Finance API

### Dataset Scale

- **Training Set**: 8,377 financial news articles
- **Validation Set**: 1,047 articles
- **Test Set**:  1,047 articles
- **Time Period**: 2020-2024

### Data Labeling

Labels are automatically generated based on **actual stock price changes** after news publication: 
- Calculate stock price change on the day of news publication
- Automatically assign labels to 7 categories based on price movement
- Ensures objectivity and accuracy of labels

---

## 🧠 Model Architecture

### Base Model

```
FinBERT (ProsusAI/finbert)
├── Based on BERT-base-uncased
├── Pre-trained on financial domain corpus
└── Parameters: 110M
```

### Fine-tuning Architecture

```python
FinBERT
├── Transformer Encoder (12 layers)
├── Pooler Layer
└── Classification Head
    ├── Dropout (0.1)
    └── Linear (768 → 7)
```

### Training Configuration

```yaml
Optimizer: AdamW
Learning Rate: 2e-5
Batch Size:  16
Epochs: 10
Max Length: 128 tokens
Loss Function: CrossEntropyLoss
Device:  CUDA (GPU)
```

---

## 🚀 Installation & Usage

### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- 24GB+ RAM

### 1. Clone Repository

```bash
git clone https://github.com/chenshi20250824-a11y/7-class-bert.git
cd finbert-sentiment-analysis
```

### 2. Create Virtual Environment

```bash
# Using Conda
conda create -n finbert python=3.8
conda activate finbert

# Or using venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model

Model files are large (~500MB). Two options:

**Option A:  Auto-download from Hugging Face**

```bash
# Model will be automatically downloaded to finbert_7class_model/
python download_model.py
```

**Option B: Manual download**

Download from [Google Drive](link) or [Baidu Netdisk](link) and extract to project root. 

### 5. Run Web Application

```bash
cd 7201_project
python app_flask.py
```

### 6. Access Website

Open browser and visit: `http://localhost:5000`

---

## 📖 User Guide

### Using the Web Interface

#### Method 1: Manual Input

1. Enter English financial news in the text box
2. Click **"🔍 Analyze"** button
3. View prediction results

#### Method 2: Use Examples

1. Click example button (e.g., "📈 Apple earnings exceed expectations")
2. News automatically fills the input box
3. Click **"🔍 Analyze"** to view results

### Interpreting Results

**Prediction results include:**
- ✅ Predicted category (e.g., RISE)
- ✅ Chinese name (e.g., 中涨)
- ✅ Expected price change (e.g., 2% ~ 5%)
- ✅ Confidence level (e.g., 78. 11%)
- ✅ Investment advice
- ✅ Probability distribution for all categories

### Example

**Input News:**
```
Pre-tax loss totaled euro 0.3 million ,compared to a loss of euro 2.2 million in the first quarter of 2005
```

**Prediction Result:**
- Category: **rise**
- Price Change: 2% ~ 5%
- Confidence:  78.11%
- Advice: 📈 Strong bullish signal, buying is recommended

---

## 📁 Project Structure

```
7201_project/
├── 📂 financial_data/              # Raw data
│   ├── AAPL_prices.csv
│   ├── alphavantage_news.csv
│   └── ... 
│
├── 📂 finbert_7class_model/        # Trained model
│   ├── config.json
│   ├── model. safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── label_mapping.json
│   └── training_summary.json
│
├── 📄 app_flask.py                 # Flask web application
├── 📄 Xiangrui_Finbert.ipynb       # Training & evaluation code
├── 📄 training_data_2025.csv       # Training dataset
├── 📄 requirements.txt             # Dependencies
├── 📄 README.md                    # This file
├── 📄 启动FinBERT网站.bat          # Quick start script (Windows)
└── 📄 . gitignore
```


## 📈 Performance Metrics

### Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | XX. X% |
| **Precision** | XX.X% |
| **Recall** | XX. X% |
| **F1 Score** | XX.X% |

### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| crash | XX% | XX% | XX% | XXX |
| drop | XX% | XX% | XX% | XXX |
| small_drop | XX% | XX% | XX% | XXX |
| stable | XX% | XX% | XX% | XXX |
| small_rise | XX% | XX% | XX% | XXX |
| rise | XX% | XX% | XX% | XXX |
| surge | XX% | XX% | XX% | XXX |

---

## ❓ FAQ

### Q1: Why are predictions sometimes inaccurate?

**A:** Possible reasons:
- News text is too short or lacks information
- News content is outside the model's training scope
- The dataset is too small
- The website's anti-scraping measures have been strengthened, making it difficult to find data.

### Q2: Can it analyze Chinese news?

**A:** Currently, the model only supports English news. For Chinese news: 
1. Use translation API to convert to English
2. Or retrain with Chinese financial model (e.g., FinBERT-Chinese)

### Q3: How to improve prediction accuracy?

**A:** Improvement directions:
- Increase training data volume
- Use larger models (e.g., FinBERT-large)
- Integrate multiple features (technical indicators, market sentiment, etc.)
- Use ensemble learning methods (GARCH-X Model,Regression model,Vector Autoregression,Volatility Measurement)

### Q4: Website cannot be accessed?

**A:** Checklist:
- [ ] Confirm server is running
- [ ] Check if port 5000 is occupied
- [ ] Check firewall settings
- [ ] Clear browser cache

### Q5: Can the model be used for actual trading?

**A:** ⚠️ **Not recommended! **
- This project is for educational and research purposes only
- Predictions are for reference only
- Investment involves risks; decisions should be made cautiously
- Combine with fundamental and technical analysis

---
## ⚠️ Limitations and Future Improvements

### Current Limitations

#### 1. Time Window Selection

- **Current Approach**: Using same-day stock price changes as labels
- **Issue**: When predicting newly published news, stock prices may not have fully reacted yet
- **Impact**:  Potential inconsistency between training and application scenarios

#### 2. Data Timeliness

- Changes in market conditions may affect model performance

#### 3. Language Constraints
- Other languages (e.g., Chinese) require retraining

### Improvement Roadmap

#### Short-term Improvements (1-3 months)

- [ ] Implement multi-time-window prediction (T+0, T+1, T+2)
- [ ] Compare effectiveness of different time windows
- [ ] Dynamically adjust prediction strategy based on news publication time

#### Mid-term Improvements (3-6 months)

- [ ] Increase training data volume to 50,000+ articles
- [ ] Support Chinese news analysis
- [ ] Integrate multi-dimensional features (technical indicators, market sentiment, etc.)

#### Long-term Vision (6-12 months)

- [ ] Real-time news stream processing
- [ ] Multi-stock correlation analysis
- [ ] Reinforcement learning for trading strategy optimization

---

## 🔮 Future Improvements

### Feature Extensions

- [ ] Support Chinese news analysis
- [ ] Add historical prediction records
- [ ] Export analysis reports (PDF)
- [ ] Integrate real-time news crawler
- [ ] Support batch analysis

### Model Optimization

- [ ] Use larger models (FinBERT-large)
- [ ] Multi-task learning (sentiment + entity recognition)
- [ ] Integrate multi-modal data (text + numerical)
- [ ] Time series modeling
- [ ] Reinforcement learning optimization

### Engineering Improvements

- [ ] Deploy to cloud servers
- [ ] Containerized deployment (Docker)
- [ ] CI/CD automation
- [ ] API development
- [ ] Performance optimization and caching

---

## 👨‍💻 Author

**[Yixiangrui]**

- 📧 Email: MC56586@um.edu.mo
- 🎓 Institution: [University Of Macau]
- 🐱 GitHub: [@chenshi20250824-a11y](https://github.com/chenshi20250824-a11y/7-class-bert.git)

---

## 🙏 Acknowledgments

### Open Source Projects

- [Hugging Face Transformers](https://github.com/chenshi20250824-a11y/7-class-bert.git)
- [FinBERT](https://github.com/ProsusAI/finBERT)
- [Flask](https://flask.palletsprojects.com/)
- [PyTorch](https://pytorch.org/)

### Data Sources

- [Alpha Vantage](https://www.alphavantage.co/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [FinViz](https://finviz.com/)

### References

```bibtex
@article{araci2019finbert,
  title={Finbert: Financial sentiment analysis with pre-trained language models},
  author={Araci, Dogu},
  journal={arXiv preprint arXiv:1908.10063},
  year={2019}
}

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ⚠️ Disclaimer

**Important Notice:**

1. This project is for **educational and research purposes only**
2. Predictions **do not constitute investment advice**
3. Investment involves risks; enter the market with caution
4. Using this project for investment decisions is **at your own risk**
5. The author is not responsible for any investment losses

---

## 📮 Contact

For questions or suggestions, please contact via: 

- 📧 Email: MC56586@um.edu.mo
- 💬 Submit [Issue](https://github.com/chenshi20250824-a11y/7-class-bert.git)
- 🔀 Submit [Pull Request](https://github.com/chenshi20250824-a11y/7-class-bert.git)

---

## ⭐ Star History

If this project helps you, please give it a Star ⭐! 

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/finbert-sentiment&type=Date)](https://star-history.com/#yourusername/finbert-sentiment&Date)

---

<div align="center">

**Made with ❤️ by [YiXiangrui]**

[⬆ Back to Top](#-FinBERT-Financial-News-Sentiment-Analysis-and-Stock-Price-Prediction-System)

</div>
