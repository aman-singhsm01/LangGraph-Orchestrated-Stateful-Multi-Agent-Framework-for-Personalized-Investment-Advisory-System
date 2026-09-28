import os
from dotenv import load_dotenv
# Load the API keys from your .env file
load_dotenv()

import pandas as pd
import numpy as np
import camelot
import joblib
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from transformers import pipeline
import chromadb

# --- Import Gemini for the final explainability ---
from langchain_google_genai import ChatGoogleGenerativeAI

# Internal Imports
from ml_models import fetch_and_preprocess_data, evaluate_models
from auditor import HybridCategorizer

# Initialize RAG Memory for Profile
chroma_client = chromadb.Client()
memory_collection = chroma_client.get_or_create_collection(name="financial_memory")

class FinancialState(TypedDict):
    pdf_path: str
    pdf_password: str
    ticker: str
    surplus: float
    market_data: Any 
    model_metrics: Dict[str, Any]
    best_model_name: str
    pred_before_sentiment: float 
    pred_after_sentiment: float  
    sentiment_score: float
    risk_profile: str

class FiscalAgent:
    def __init__(self, pdf_path, password):
        self.pdf_path = pdf_path
        self.password = password
        self.categorizer = HybridCategorizer()
        if not os.path.exists('tx_model.pkl'):
            self.categorizer.train_with_split("C:/Users/amans/Desktop/Finalyear_run/Personal_Finance_Dataset.csv")
        else:
            self.categorizer.model = joblib.load('tx_model.pkl')
            self.categorizer.vectorizer = joblib.load('vectorizer.pkl')

    def clean_currency(self, value):
        if pd.isna(value) or str(value).strip() == "": return 0.0
        cleaned = str(value).replace(',', '').replace(' ', '').replace('₹','').strip()
        try: return float(cleaned)
        except: return 0.0

    def audit_finances(self):
        print("\n--- [Stage 1] Intelligent Fiscal Auditing  ---")
        try:
            tables = camelot.read_pdf(self.pdf_path, password=self.password, pages='all', flavor='lattice')
            if tables.n == 0:
                tables = camelot.read_pdf(self.pdf_path, password=self.password, pages='all', flavor='stream')

            df = pd.concat([table.df for table in tables]).reset_index(drop=True)
            
            spending_log = {
                "Grocery": 0.0, 
                "Food": 0.0, 
                "Transport": 0.0, 
                "Bills": 0.0, 
                "Healthcare": 0.0, 
                "Miscellaneous": 0.0
            }
            
            total_deposits = 0.0
            total_withdrawals = 0.0
            table_started = False 
            
            print("Running Hybrid Classifier on Bank Statement...")
            for _, row in df.iterrows():
                row_str = " ".join([str(x).lower() for x in row.values])
                
                if "transaction overview" in row_str:
                    table_started = True
                    continue
                
                if not table_started:
                    if len(row) >= 5 and self.clean_currency(row.iloc[-3]) > 0:
                        table_started = True
                    else:
                        continue
                        
                if len(row) < 5: continue 
                
                raw_desc = str(row.iloc[1]).replace('\n', ' ').strip() if len(row) > 1 else ""
                
                if len(raw_desc) < 3 or not any(c.isalpha() for c in raw_desc) or "balance" in raw_desc.lower():
                    continue
                
                if "/" in raw_desc:
                    parts = raw_desc.split("/")
                    clean_parts = [p.strip() for p in parts if p.strip()]
                    if len(clean_parts) >= 2:
                        desc = f"{clean_parts[-2]} {clean_parts[-1]}"
                    else:
                        desc = clean_parts[-1] if clean_parts else raw_desc
                else:
                    desc = raw_desc
                
                deposit = self.clean_currency(row.iloc[-3])
                withdrawal = self.clean_currency(row.iloc[-2])
                
                if withdrawal > 0:
                    total_withdrawals += withdrawal
                    try:
                        result = self.categorizer.classify(desc)
                        
                        if isinstance(result, tuple) and len(result) >= 1:
                            cat = result[0]
                        elif isinstance(result, list) and len(result) >= 1:
                            cat = result[0]
                        elif isinstance(result, str):
                            cat = result
                        else:
                            cat = "Miscellaneous"
                            
                        cat = str(cat).strip().title()
                        if cat not in spending_log:
                            cat = "Miscellaneous"
                    except Exception:
                        cat = "Miscellaneous"
                    
                    spending_log[cat] += withdrawal
                        
                if deposit > 0:
                    total_deposits += deposit

            surplus = total_deposits - total_withdrawals
            self.categorizer.generate_report(spending_log, total_deposits, surplus)
            return surplus, spending_log
        except Exception as e:
            print(f"Audit Error: {e}")
            return 0.0, {}

# --- Node Functions ---

def fiscal_node(state: FinancialState):
    agent = FiscalAgent(state["pdf_path"], state["pdf_password"])
    surplus, _ = agent.audit_finances()
    u_input = input(f"\nAI Suggested Surplus: ₹{surplus:.2f}. Confirm (y) or Enter Manual: ").strip().lower()
    return {"surplus": float(u_input) if u_input.replace('.','',1).isdigit() else surplus}

def market_node(state: FinancialState):
    print("\n--- [Stage 2] Market Intelligence: 10-Model Prediction ---")
    df = fetch_and_preprocess_data(state["ticker"])
    

    df = pd.DataFrame(df.to_numpy(), index=df.index, columns=df.columns).astype('float64')
    
    pred_before, metrics, best_model = evaluate_models(df)
    
    print("\n--- [Stage 2.5] FinBERT Sentiment Analysis ---")
    try:
        sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        res = sentiment_pipe([f"{state['ticker']} stock shows growth potential and strong institutional interest."])[0]
        score = res['score'] if res['label'] == 'positive' else (-res['score'] if res['label'] == 'negative' else 0.0)
    except Exception as e:
        print(f"Sentiment Error: {e}")
        score = 0.0
    
    return {
        "market_data": df, "model_metrics": metrics, "best_model_name": best_model,
        "pred_before_sentiment": pred_before, "pred_after_sentiment": pred_before * (1 + (score * 0.02)),
        "sentiment_score": score
    }

def advisor_node(state: FinancialState):
    risk = input("\n--- [Stage 3] Risk Alignment ---\nEnter Risk Profile (Low/Medium/High): ").strip().capitalize()
    return {"risk_profile": risk}

def output_node(state: FinancialState):
    print("\n--- [Stage 4] Final Synthesis & Explainability ---")
    
    curr_price = state["market_data"]['Close'].iloc[-1]
    base_pred = state["pred_before_sentiment"]
    final_pred = state["pred_after_sentiment"]
    growth = ((final_pred - curr_price) / curr_price) * 100
    sentiment = state["sentiment_score"]
    
    risk_map = {"Low": 3.0, "Medium": 2.0, "High": 1.0}
    threshold = risk_map.get(state["risk_profile"], 2.0)
    
    # --- NEW: Unit Calculation Logic ---
    units_affordable = int(state["surplus"] // curr_price) if state["surplus"] > 0 else 0
    total_investment_cost = units_affordable * curr_price
    
    # Enhanced Decision Logic
    if growth > threshold and units_affordable > 0:
        decision = "BUY"
        action_text = f"BUY {units_affordable} Units (Costing ₹{total_investment_cost:.2f})"
    elif growth < 0:
        decision = "SELL"
        action_text = "SELL / AVOID (Negative Growth Expected)"
    elif units_affordable == 0 and growth > threshold:
        decision = "HOLD"
        action_text = "HOLD (Insufficient Surplus to Buy 1 Unit)"
    else:
        decision = "HOLD"
        action_text = f"HOLD (Can afford {units_affordable} units, but growth doesn't meet {state['risk_profile']} risk threshold)"

    # 1. Print the Exact Math
    print(f"\n" + "="*80)
    print(f" FINAL RECOMMENDATION: {decision} {state['ticker']}")
    print("="*80)
    print(f" [NUMERIC BREAKDOWN]")
    print(f" • Current Stock Price             : ₹{curr_price:.2f}")
    print(f" • Base ML Prediction ({state['best_model_name']:<12}) : ₹{base_pred:.2f}")
    print(f" • FinBERT Sentiment Confidence    :  {sentiment:.4f} ({'BULLISH' if sentiment > 0 else 'BEARISH'})")
    print(f" • Final Adjusted Prediction       : ₹{final_pred:.2f} ({growth:+.2f}% Expected Growth)")
    print(f" • Required Growth for {state['risk_profile']} Risk :  {threshold:+.2f}%")
    print(f" • Available Investable Surplus    : ₹{state['surplus']:.2f}")
    print(f" • Purchasing Power                : {units_affordable} Units")
    print(f" • Recommended Action              : {action_text}")
    print("="*80)
    
    # 2. Generate LLM Explainability
    print("\nGenerating LLM Explainability Report...\n")
    try:
        # Using Gemini's stable model endpoint
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        prompt = f"""
        You are the final explainability agent in a highly advanced multi-agent financial system. 
        Explain the logic behind the following algorithmic trading decision to the user in 3-4 clear, professional bullet points.
        
        Data points:
        - Ticker: {state['ticker']}
        - Action: {action_text}
        - Current Price: ₹{curr_price:.2f}
        - ML Predicted Price: ₹{base_pred:.2f} (Algorithm: {state['best_model_name']})
        - Market Sentiment Impact: Shifted to ₹{final_pred:.2f} (Confidence: {sentiment:.4f})
        - User's Available Surplus: ₹{state['surplus']:.2f}
        - User's Risk Profile: {state['risk_profile']} (Requires >{threshold}% growth)
        - Expected Growth: {growth:.2f}%
        
        Keep the tone objective, analytical, and highly professional. Focus purely on *why* these numbers led to the {decision} action. Be sure to mention the specific number of units recommended.
        """
        explanation = llm.invoke(prompt).content
        print("[AI EXPLAINABILITY JUSTIFICATION]")
        print(explanation)
        print("="*80 + "\n")
    except Exception as e:
        print(f"Explainability Generation Failed: {e}")
        
    return state

def build_graph():
    workflow = StateGraph(FinancialState)
    workflow.add_node("fiscal_auditor", fiscal_node)
    workflow.add_node("market_analyst", market_node)
    workflow.add_node("risk_advisor", advisor_node)
    workflow.add_node("output_agent", output_node)
    
    workflow.set_entry_point("fiscal_auditor")
    workflow.add_edge("fiscal_auditor", "market_analyst")
    workflow.add_edge("market_analyst", "risk_advisor")
    workflow.add_edge("risk_advisor", "output_agent")
    workflow.add_edge("output_agent", END)
    
    return workflow.compile()