# Book Recommendation System

A comprehensive book recommendation system that uses Google Books API to collect book data, creates embeddings using OpenAI, and provides semantic search-based recommendations using FAISS vector store.

## 🏗️ System Architecture

### Overview

The system consists of three main components:

1. **Data Pipeline** (`src/data-pipeline.py`) - Collects and processes book data
2. **Recommendation Service** (`src/recommendation_service.py`) - Provides book recommendations
3. **Vector Store** - FAISS-based similarity search engine

### Architecture Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Google Books  │    │   OpenAI API    │    │   FAISS Vector  │
│      API        │───▶│   Embeddings    │───▶│     Store       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Pipeline  │    │  Recommendation │    │  User Interface │
│  (Collection &  │    │     Service     │    │   (Query &      │
│   Processing)   │    │  (Similarity    │    │   Results)      │
└─────────────────┘    │    Search)      │    └─────────────────┘
                       └─────────────────┘
```

## 📊 Data Pipeline

### Components

#### 1. SystematicGoogleBooksLoader
- **Purpose**: Fetches book data from Google Books API systematically
- **Features**:
  - Multiple search strategies (subject, author, title)
  - Rate limiting with exponential backoff
  - Duplicate prevention
  - Quality filtering (minimum description length)

#### 2. DataProcessor
- **Purpose**: Cleans and preprocesses collected data
- **Operations**:
  - Removes duplicates
  - Handles missing values
  - Normalizes text (whitespace, formatting)
  - Filters empty descriptions

#### 3. Embedding Pipeline
- **Purpose**: Creates vector embeddings for semantic search
- **Process**:
  - Uses OpenAI's text-embedding-3-small model
  - Batch processing for efficiency
  - Stores embeddings with metadata

### Data Flow
```
Google Books API → Raw Data → Cleaning → Embeddings → Vector Store
```

### Search Strategies
The system uses three search strategies to collect diverse book data:

1. **Subject-based**: Fiction, Science, History, Biography, Technology, Art, Philosophy, Psychology, Business, Travel
2. **Author-based**: Stephen King, J.K. Rowling, Yuval Noah Harari, Michelle Obama, Malcolm Gladwell, J.R.R. Tolkien, Agatha Christie, Toni Morrison
3. **Title-based**: Python, Machine Learning, World War, Artificial Intelligence, Space Exploration, Climate Change, Ancient History, Mental Health

## 🔍 Recommendation Service

### Features
- **Semantic Search**: Uses FAISS vector similarity search
- **Relevance Scoring**: Returns similarity scores with recommendations
- **Flexible Querying**: Accepts natural language queries
- **Configurable Results**: Adjustable number of recommendations (k)

### How It Works

1. User provides a natural language query
2. Query is embedded using the same OpenAI model
3. FAISS performs similarity search against book embeddings
4. Results are ranked by similarity score
5. Recommendations are returned with metadata

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- Google Books API key
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Book-recommendation-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_books_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

### API Keys Setup

#### Google Books API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Books API
4. Create credentials (API Key)
5. Add the key to your `.env` file

#### OpenAI API
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account and get API key
3. Add the key to your `.env` file

## 🏃‍♂️ Running the System

### 1. Data Collection and Processing

**Option A: Run Complete Pipeline (API calls)**
```bash
cd src
python data-pipeline.py
```

**Option B: Use Existing Data (if you have books.csv)**
The pipeline is configured to use existing data by default. If you have `data/books.csv`, it will skip the API collection phase.

### 2. Get Book Recommendations

```bash
cd src
python recommendation_service.py
```

### 3. Custom Queries

You can modify the query in `recommendation_service.py`:

```python
if __name__ == "__main__":
    query = "your custom query here"  # Change this
    results = recommend_books(query, 5)  # Adjust number of results
    visualize_results(results)
```

## 📁 Project Structure

```
Book-recommendation-system/
├── README.md
├── requirements.txt
├── setup.sh
├── .env                    # Environment variables (create this)
├── data/
│   ├── books.csv          # Raw book data
│   └── books_embeddings.csv # Processed data with embeddings
├── src/
│   ├── data-pipeline.py   # Data collection and processing
│   ├── recommendation_service.py # Recommendation engine
│   └── interface.py       # User interface (placeholder)
├── book_index/            # FAISS vector store (auto-generated)
└── notebooks/             # Jupyter notebooks for analysis
```

## ⚙️ Configuration

### Rate Limiting
The system includes built-in rate limiting to respect API limits:
- 2-second delay between successful requests
- Exponential backoff for 429 errors (up to 5 minutes)
- Maximum 3 retries per query

### Data Quality Filters
- Minimum description length: 2048 characters
- Removes duplicate books
- Filters out books with missing titles or descriptions

### Embedding Settings
- Model: `text-embedding-3-small`
- Batch size: 50 documents
- Vector dimensions: 1536

## 📊 Data Schema

The system collects the following book information:

| Field | Description |
|-------|-------------|
| `id` | Unique Google Books ID |
| `title` | Book title |
| `authors` | Author names (comma-separated) |
| `description` | Book description/summary |
| `categories` | Book categories (comma-separated) |
| `primary_category` | Main category |
| `publisher` | Publisher name |
| `published_date` | Publication date |
| `page_count` | Number of pages |
| `maturity_rating` | Age rating |
| `isbn_10` | ISBN-10 |
| `isbn_13` | ISBN-13 |
| `language` | Book language |
| `search_strategy` | How the book was found |
| `retrieval_timestamp` | When data was collected |
| `embedding` | Vector representation |

## 🔧 Troubleshooting

### Common Issues

1. **429 Too Many Requests**
   - The system includes exponential backoff
   - Wait for the retry mechanism to complete
   - Consider reducing search strategies if needed

2. **API Key Errors**
   - Verify your API keys are correctly set in `.env`
   - Ensure you have sufficient API quota

3. **Memory Issues**
   - Reduce batch size in embedding process
   - Process data in smaller chunks

4. **File Not Found Errors**
   - Ensure `data/` directory exists
   - Check file paths in the code

### Performance Tips

1. **For Large Datasets**
   - Use smaller batch sizes
   - Process data incrementally
   - Consider using GPU-accelerated FAISS

2. **For Production**
   - Implement proper logging
   - Add monitoring for API quotas
   - Consider caching mechanisms