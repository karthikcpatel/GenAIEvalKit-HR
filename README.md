# 🧠 # GenAIEvalKit-HR  
**GenAI Evaluation Framework for HR Chatbots**

A modular, production-ready framework for testing and evaluating Generative AI (LLM) systems, including RAG pipelines, hallucination detection, safety checks, and LLM-as-a-judge scoring.

---

## 🚀 Features

- ✅ Semantic similarity evaluation (embeddings)
- ✅ Token-level metrics (Precision, Recall, F1)
- ✅ Hallucination detection
- ✅ Toxicity / safety checks
- ✅ LLM-as-a-judge evaluation
- ✅ RAG (Retrieval-Augmented Generation) pipeline
- ✅ Regression testing
- ✅ Latency / performance testing
- ✅ Allure reporting support
- ✅ Custom HTML report generation

---

## 📂 Project Structure
gen-ai-framework/
│
├── evaluator/
│ ├── similarity.py
│ ├── metrics.py
│ ├── hallucination.py
│ └── toxicity.py
│
├── judge/
│ ├── judge_prompts.py
│ └── llm_judge.py
│
├── rag/
│ ├── rag_pipeline.py
│ └── retriever.py
│
├── llm/
│ └── llm_client.py
│
├── prompts/
│ └── hr_prompt_suite.csv
│
├── tests/
│ ├── test_hallucination.py
│ ├── test_metrics.py
│ ├── test_toxicity.py
│ ├── test_regression.py
│ ├── test_latency.py
│ ├── test_llm_judge.py
│ └── test_rag_grounding.py
│
├── reports/
│ └── report.html
│
├── run_tests.py
├── requirements.txt
└── README.md


---

## ⚙️ Setup

### 1. Clone repository
```bash
git clone <repo-url>
cd gen-ai-framework
2. Install dependencies
pip install -r requirements.txt
3. Configure environment variables

Create .env file:

LLM_API_KEY=your_api_key_here
4. (Optional) Create .env.example
LLM_API_KEY=your_api_key_here
🧪 Running Tests
Run all tests (excluding performance)
python run_tests.py
Run integration tests only
pytest -m integration -v
Run performance tests
pytest -m performance -v
Run with Allure report
pytest --alluredir=reports/allure-results
allure serve reports/allure-results
🧠 Evaluation Capabilities
1. Hallucination Detection
Based on semantic similarity
Configurable thresholds
Explanation included
2. Metrics (PRF)
Token-based precision, recall, F1
Frequency-aware scoring
Optional stopword removal
3. Toxicity Detection
Keyword + weighted scoring
Threshold-based classification
Safe regex matching
4. LLM-as-a-Judge
Scores:
correctness
completeness
safety
Structured JSON output
Robust parsing with retries
5. RAG Evaluation
Semantic retrieval from dataset
Grounded answer generation
Context-based validation
6. Regression Testing
Detects model drift
Uses similarity thresholds instead of exact match
7. Latency Testing
Measures:
average latency
max latency
p95 latency
SLA validation
📊 Dataset

Located at:

prompts/hr_prompt_suite.csv

Includes:

HR-related queries
Expected answers
Categories (leave, PF, mediclaim, general)
Key points
Difficulty levels
📈 Reporting
1. Allure Reports
Detailed step-by-step execution
Best for debugging
2. HTML Report

Generated via:

generate_report(results)

Outputs:

reports/report.html