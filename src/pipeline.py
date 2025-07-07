import os
from langchain.chains import SequentialChain
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from core.data_loader import GoogleBooksLoader
from chains.processing_chains import (
    DataCleaningChain, 
    TextPreprocessingChain, 
    EmbeddingGenerationChain, 
    VectorStoreCreationChain
)

books_file = "../data/books.csv"

def create_automated_data_pipeline():
    """Create a complete automated data pipeline using chains"""
    
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create individual chains
    cleaning_chain = DataCleaningChain()
    preprocessing_chain = TextPreprocessingChain(config={
        'remove_stopwords': True,
        'use_lemmatization': True,
        'remove_punctuation': True,
        'normalize_case': True,
        'min_word_length': 3
    })
    embedding_chain = EmbeddingGenerationChain(embeddings_model)
    vector_store_chain = VectorStoreCreationChain(embeddings_model)
    
    full_pipeline = SequentialChain(
        chains=[cleaning_chain, preprocessing_chain, embedding_chain, vector_store_chain],
        input_variables=["raw_data"],
        output_variables=["cleaned_data", "preprocessed_data", "embedded_data", "vector_store_path"],
        verbose=True
    )
    
    return full_pipeline

def collect_data():
    """Collect new data from Google Books API"""
    try:
        loader = GoogleBooksLoader()
        raw_data = loader.load()
        
        if raw_data is None or raw_data.empty:
            print("No data collected")
            return None
            
        print(f"Successfully collected {len(raw_data)} books")
        return raw_data
        
    except Exception as e:
        print(f"Error collecting data: {str(e)}")
        return None
    
def run_data_pipeline(books_file):
    """Run the complete data processing pipeline"""
    
    if not os.path.exists(books_file):
        print("No existing data found. Collecting new data...")
        raw_data = collect_data()
        if raw_data is None:
            return None
        raw_data.to_csv(books_file, index=False)
    else:
        print("Loading existing data...")
        raw_data = pd.read_csv(books_file)
    
    pipeline = create_automated_data_pipeline()
    results = pipeline.invoke({"raw_data": raw_data})
    
    print("Pipeline completed!")
    print(f"Cleaned data shape: {results['cleaned_data'].shape}")
    print(f"Preprocessed data shape: {results['preprocessed_data'].shape}")
    print(f"Embedded data shape: {results['embedded_data'].shape}")
    print(f"Vector store created at: {results['vector_store_path']}")
    
    return results

if __name__ == "__main__":
    run_data_pipeline(books_file) 