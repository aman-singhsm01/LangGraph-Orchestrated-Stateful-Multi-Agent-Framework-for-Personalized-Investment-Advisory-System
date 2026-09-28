import warnings
warnings.filterwarnings("ignore", module="requests")
warnings.filterwarnings("ignore", category=UserWarning)
import os
from dotenv import load_dotenv
from agents import build_graph

# Load environment variables (Tavily, HF keys if needed later)
load_dotenv()

if __name__ == "__main__":
    print("Initializing Multi-Agent Financial Intelligence System...")
    
    # Build and compile the LangGraph workflow
    app = build_graph()
    
   
    initial_state = {
        "pdf_path": "bank_statement_may2025 (2).pdf",
        "pdf_password":"33242050502", 
        "ticker": "AAPL",                     
        "surplus": 0.0,
        "market_data": None,
        "model_metrics": {},
        "final_pred": 0.0,
        "sentiment_score": 0.0,
        "risk_profile": ""
    }
    
    try:
        # Execute the multi-agent workflow
        result = app.invoke(initial_state)
        print("\n--- Workflow Execution Completed Successfully ---")
    except Exception as e:
        print(f"\nAn error occurred during workflow execution: {e}")