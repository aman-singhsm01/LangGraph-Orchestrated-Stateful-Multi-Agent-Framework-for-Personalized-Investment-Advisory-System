# LangGraph-Orchestrated-Stateful-Multi-Agent-Framework-for-Personalized-Investment-Advisory-System
LangGraph-Orchestrated Stateful Multi-Agent Framework for Personalized Investment Advisory System Using Hybrid Forecasting and LLM Explainability
<div align="center">

# 🧠 Multi-Agent Financial Intelligence System
**Bridging Global Market Intelligence with Personal Fiscal Reality via LangGraph, Deep Learning, and LLMs**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-FF6F00?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gemini AI](https://img.shields.io/badge/Google_Gemini-Explainable_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![HuggingFace](https://img.shields.io/badge/FinBERT-Sentiment_NLP-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ProsusAI/finbert)

</div>

---

## 📌 Project Overview

Effective investment management is heavily hindered by the disconnect between **global market intelligence** and **personal fiscal reality**. Traditional quantitative frameworks focus purely on market-centric price action while remaining blind to an investor's personal liquidity, spending habits, and risk thresholds.

The **Multi-Agent Financial Intelligence System** is an end-to-end personalized investment advisory pipeline[cite: 2, 5]. Coordinated through a stateful **LangGraph** architecture, the system ingests encrypted personal bank statements to calculate real-time investable surplus, evaluates 5 years of historical equity data across a **10-model ML/DL ensemble** enhanced with Wavelet denoising, adjusts price targets via **FinBERT** market sentiment, and synthesizes human-readable **BUY / HOLD / SELL** recommendations with exact unit affordability using **Google Gemini**[cite: 2, 6].

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Stage 1: Intelligent Fiscal Auditing
        A[📄 Encrypted PDF Bank Statement] -->|Camelot Lattice/Stream| B(Raw Transaction Extraction)
        B --> C{3-Tier Hybrid Categorizer}
        C -->|Tier 1: Exact Match| D[Rule-Based Keyword Mapping]
        C -->|Tier 2: Prob > 0.5| E[TF-IDF + SMOTE Random Forest]
        C -->|Tier 3: Fallback| F[Gemini 1.5 Flash LLM]
        D & E & F --> G[(ChromaDB Vector Memory)]
        G --> H[💰 Calculate Investable Surplus & Pie Chart]
    end

    subgraph Stage 2: Market Intelligence & Sentiment
        I[📈 Yahoo Finance 5Y OHLCV] --> J[Feature Engineering: RSI, MACD, Signal]
        J --> K[Wavelet Denoising: Daubechies db4]
        K --> L[10-Model Training & Evaluation]
        L --> M1[Time-Series: ARIMA]
        L --> M2[Deep Learning: LSTM, GRU, BiLSTM]
        L --> M3[ML Regressors: RF, XGBoost, AdaBoost, DT, LR]
        M1 & M2 & M3 --> N[Dynamic Top-3 Hybrid Ensemble]
        N -->|Select Lowest MSE| O[Base Price Prediction]
        P[📰 Financial News / Context] -->|ProsusAI/FinBERT| Q[Sentiment Score Confidence]
        O & Q --> R[Sentiment-Adjusted Price Target]
    end

    subgraph Stage 3 & 4: Risk Alignment & Explainability
        H & R --> S{Risk Profile Alignment}
        S -->|Low: >3% / Med: >2% / High: >1%| T[Unit Purchasing Power & Decision Engine]
        T --> U[🤖 Gemini 2.5 Flash Explainability Agent]
        U --> V([🎯 Final BUY / HOLD / SELL Report])
    end
