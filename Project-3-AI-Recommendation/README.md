# DecodeLabs AI Internship — Project 3
# AI Recommendation Logic — Tech Stack Recommender 

A content-based recommendation engine that maps a user's skills to the most
relevant tech career paths, built as the third milestone of the DecodeLabs
Artificial Intelligence Internship (Batch 2026).

# Objective
Build a recommendation system using TF-IDF feature extraction and Cosine
Similarity to output the Top-3 most relevant job roles for a user's skill set.

# Features
- 15 job roles × 8 skills each (curated dataset: `raw_skills.csv`)
- TF-IDF vectorization penalizes generic skills (e.g., Git) and rewards specific ones (e.g., Solidity)
- Cosine Similarity for magnitude-invariant matching
- 4-step pipeline: Ingestion → Scoring → Sorting → Filtering
- Top-3 ranked output with match percentages and visual bars
- Interactive loop — run multiple queries without restarting

# Concepts Applied (from Project PDF)
- **IPO Model**: Input (skills) → Process (TF-IDF + Cosine) → Output (Top-3)
- **Content-Based Filtering**: Matches based on item attributes, not user history
- **Vector Mapping**: Text → numerical vectors in shared vocabulary space
- **TF-IDF**: Term Frequency × Inverse Document Frequency for weighted features
- **Cosine Similarity**: Angle-based (not distance-based) similarity
- **Cold Start Bypass**: Enforces minimum 3 user inputs for data density
- **Choice Overload Prevention**: Top-N filtering

# How to Run
```bash
pip install -r requirements.txt
python recommender.py
