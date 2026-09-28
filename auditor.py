import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import chromadb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class HybridCategorizer:
    def __init__(self, model_path='tx_model.pkl', vec_path='vectorizer.pkl'):
        self.model_path = model_path
        self.vec_path = vec_path
        
        if os.path.exists(self.model_path) and os.path.exists(self.vec_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vec_path)
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced',
                min_samples_leaf=2
            )
            
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

        self.chroma_client = chromadb.PersistentClient(path="./finance_memory")
        self.collection = self.chroma_client.get_or_create_collection(name="transaction_history")

    def train_with_split(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"Error: Dataset not found at {csv_path}")
            return
        print(f"\n--- [Auditor] Training Classifier on {csv_path} ---")
        df = pd.read_csv(csv_path)

        # ✅ FIX 1: mapping is now actually applied via .replace()
        mapping = {
            "Rent": "Housing",
            "Food & Drink": "Food",
            "Dining": "Food",
            "Groceries": "Food",
            "Investment": "Savings",
            "Shopping": "Shopping",
            "Entertainment": "Entertainment",
            "Utilities": "Utilities",
            "Transport": "Transportation",
            "Medical": "Healthcare"
        }
        df['Category'] = df['Category'].replace(mapping)

        valid_cats = ["Housing", "Food", "Transportation", "Utilities",
                      "Healthcare", "Debt Payments", "Savings",
                      "Shopping", "Entertainment", "Other"]
        df['Category'] = df['Category'].apply(lambda x: x if x in valid_cats else "Other")

        # Debug: show category distribution before training
        print("\nCategory distribution after mapping:")
        print(df['Category'].value_counts())
        print()

        X = df['Transaction Description'].values.astype('U')
        y = df['Category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec  = self.vectorizer.transform(X_test)

        # ✅ FIX 2: SMOTE applied correctly after vectorization
        sm = SMOTE(random_state=42, k_neighbors=2)
        X_train_balanced, y_train_balanced = sm.fit_resample(X_train_vec, y_train)

        # ✅ FIX 3: model is re-initialized with balanced params, then FIT before dump
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            min_samples_leaf=2
        )
        self.model.fit(X_train_balanced, y_train_balanced)

        # ✅ FIX 4: dump AFTER training (was happening before fit previously)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vec_path)
        print("Model and vectorizer saved.")

        # Evaluate on original (non-SMOTE) test set
        y_pred = self.model.predict(X_test_vec)
        print("\n" + "="*50)
        print(f"{'[ML Model Evaluation Results]':^50}")
        print("="*50)
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("="*50 + "\n")

        self.collection.add(
            documents=X_train.tolist()[:100],
            metadatas=[{"category": cat} for cat in y_train.tolist()[:100]],
            ids=[f"seed_{i}" for i in range(min(100, len(X_train)))]
        )

    def classify(self, text):
        try:
            text_str = str(text).strip().lower()
            if not text_str:
                return "Miscellaneous", 0.0

            keyword_map = {
                "grocery": ["zepto", "blinkit", "instamart", "grocery", "dairy", "supermarket",
                            "mart", "store", "bigbasket", "dmart", "reliance smart", "d-mart"],
                "food": ["zomato", "swiggy", "kfc", "mcdonald", "restaurant", "cafe",
                         "food", "bakery", "dining", "eatery"],
                "transport": ["irctc", "uber", "ola", "rapido", "metro", "makemytrip", "petrol",
                              "fuel", "indian oil", "bharat petroleum", "hpcl", "flight",
                              "ticket", "travel", "namma metro"],
                "bills": ["airtel", "jio", "vi", "vodafone", "bescom", "electricity", "recharge",
                          "broadband", "water", "gas", "dth", "bill", "utility", "postpaid",
                          "prepaid", "netflix", "spotify", "subscription"],
                "healthcare": ["medplus", "apollo", "pharmacy", "hospital", "clinic",
                               "diagnostics", "health", "practo", "medical", "cult.fit"],
                "miscellaneous": ["emi", "loan", "bajaj", "muthoot", "finance", "credit card",
                                  "kreditbee", "repayment", "zerodha", "groww", "upstox",
                                  "mutual", "fd", "rd", "sip", "lic", "insurance",
                                  "angelone", "investment", "amazon", "lenskart", "shopping"]
            }

            for category, keywords in keyword_map.items():
                if any(kw in text_str for kw in keywords):
                    return category.title(), 1.0

            vec = self.vectorizer.transform([text_str])

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(vec)
                if probs is not None and len(probs) > 0 and len(probs[0]) > 0:
                    best_idx = probs[0].argmax()
                    confidence = probs[0][best_idx]
                    if confidence > 0.5 and len(self.model.classes_) > best_idx:
                        return self.model.classes_[best_idx], confidence
            else:
                pred = self.model.predict(vec)
                if pred is not None and len(pred) > 0:
                    return pred[0], 1.0

            # Fallback to Gemini LLM
            prompt = (
                f"Categorize this bank transaction into exactly one of these: "
                f"Grocery, Food, Transport, Bills, Healthcare, Miscellaneous. "
                f"Transaction description: '{text_str}'. Return ONLY the category name."
            )
            response = self.llm.invoke(prompt).content.strip().title()

            valid_categories = ["Grocery", "Food", "Transport", "Bills", "Healthcare", "Miscellaneous"]
            for cat in valid_categories:
                if cat.lower() in response.lower():
                    return cat, 0.9

            return "Miscellaneous", 0.5

        except Exception:
            return "Miscellaneous", 0.0

    def generate_report(self, spending_log, total_deposits, surplus):
        print("\n" + "+" + "-"*58 + "+")
        print("|" + "DETAILED FISCAL EXPENDITURE REPORT".center(58) + "|")
        print("+" + "-"*58 + "+")
        print(f"| {'Category':<25} | {'Amount (₹)':>28} |")
        print("+" + "-"*58 + "+")

        for cat, amt in spending_log.items():
            print(f"| {cat.title():<25} | ₹{amt:>27.2f} |")

        print("+" + "-"*58 + "+")
        print(f"| {'TOTAL CREDITS (INCOME)':<25} | ₹{total_deposits:>27.2f} |")
        print(f"| {'TOTAL EXPENDITURE':<25} | ₹{(total_deposits - surplus):>27.2f} |")
        print(f"| {'INVESTABLE SURPLUS':<25} | ₹{surplus:>27.2f} |")
        print("+" + "-"*58 + "+")

        plot_data = {k: v for k, v in spending_log.items() if v > 0}

        if plot_data:
            plt.figure(figsize=(10, 6))
            plt.pie(plot_data.values(), labels=plot_data.keys(), autopct='%1.1f%%',
                    startangle=140, shadow=True)
            plt.title("Expense Distribution (Categorized by Hybrid Agent)")
            plt.axis('equal')
            plt.show()
            plt.close('all')