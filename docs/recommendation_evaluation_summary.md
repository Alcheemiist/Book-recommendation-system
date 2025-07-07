# Book Recommendation System: Evaluation Summary

## System Architecture & Process

The Book Recommendation System is designed with a modular architecture:

- **Data Collection:** Gathers book data from the Google Books API.
- **Data Processing Pipeline:** Cleans and preprocesses text, generates embeddings (OpenAI), and builds a FAISS vector store for fast similarity search. This pipeline is orchestrated using LangChain.
- **Semantic Search & Recommendation:** User queries are matched to books using semantic similarity search over the vector store.
- **User Interface:** Built with Streamlit for interactive exploration, recommendations, and visualization.

**End-to-End Process:**
1. Collect book data from the API.
2. Clean and preprocess the data (NLTK for tokenization, lemmatization, stopword removal).
3. Generate embeddings for book descriptions.
4. Build a FAISS vector index for fast semantic search.
5. Accept user queries, preprocess if enabled, and retrieve the most relevant books.

---

## Semantic Search Evaluation Results

The system was evaluated using 10 diverse queries, comparing two modes:
- **With NLTK preprocessing** (query text is cleaned and normalized)
- **Without preprocessing** (raw query text)

### Key Metrics
| Mode                   | Avg. Similarity Score | Unique Titles | Total Recommendations |
|------------------------|----------------------|---------------|-----------------------|
| With Preprocessing     | 0.252                | 50            | 50                    |
| Without Preprocessing  | 0.211                | 50            | 50                    |

- **Top recommended categories** include Business & Economics, Science, General, and Fiction.
- **Top recommended authors** include Ray Bradbury, Philip K. Dick, and others.
---
*For details, see the full reports: `recommendation_performance_report_nltk.txt` and `recommendation_performance_report_raw.txt`.* 

## Insights about the implemented improvement of data preprocessing:

- **Preprocessing user queries with NLTK** (normalization, tokenization, lemmatization, stopword removal) leads to higher average similarity scores and slightly better semantic matching.
- Both modes return a diverse set of recommendations, but preprocessing improves the quality and relevance of results.
- The most effective improvement implemented so far is the use of NLTK-based preprocessing for both the dataset and user queries.

---

## Suggested improvement : Hybrid Similarity Search
- **Combine content and metadata:** Using not just the description, but also title, author, categories, and user tags in the embedding.
- **Weighted similarity:** Assign different weights to different fields (e.g., 70% description, 20% title, 10% author).

---

