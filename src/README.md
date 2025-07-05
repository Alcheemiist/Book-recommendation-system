# Book Recommendation System - Restructured

## New Project Structure

```
src/
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── base_chain.py       # Base chain with monitoring
│   ├── text_preprocessor.py # Text preprocessing utilities
│   ├── data_loader.py      # Google Books API loader
│   └── data_processor.py   # Data processing utilities
├── chains/                  # Processing chains
│   ├── __init__.py
│   └── processing_chains.py # All processing chains
├── services/                # Business logic services
│   ├── __init__.py
│   └── recommendation_service.py # Recommendation service
├── pipeline.py              # Main pipeline orchestration
├── main.py                  # Entry point
├── interface.py             # Streamlit interface (unchanged)
└── README.md               # This file
```

## Key Improvements

### 1. **Clean Architecture**
- Separated concerns into logical modules
- Removed underscore prefixes from filenames
- Clear separation between core, chains, and services

### 2. **Fixed Import Issues**
- All imports now use proper relative paths
- No more "no known parent package" errors
- Consistent module structure

### 3. **Removed Duplicate Code**
- Consolidated text preprocessing into single module
- Unified data processing functions
- Cleaner chain implementations

### 4. **Better Error Handling**
- Automatic NLTK data download
- Proper exception handling
- Clear error messages

### 5. **Improved Maintainability**
- Each module has a single responsibility
- Clear interfaces between components
- Better documentation

## Usage

### Run the Pipeline
```bash
python -m src.main
```

### Run Data Collection
```python
from src.pipeline import collect_data
collect_data()
```

### Get Recommendations
```python
from src.services.recommendation_service import recommend_books
results = recommend_books("machine learning", 5)
```

### Test Vector Store
```python
from src.services.recommendation_service import test_vector_store
test_vector_store()
```

## Environment Variables

Make sure you have these environment variables set:
- `OPENAI_API_KEY`: Your OpenAI API key
- `GOOGLE_API_KEY`: Your Google Books API key (optional, for data collection)

## Dependencies

All dependencies are listed in `requirements.txt`. The system automatically downloads required NLTK data on first run. 