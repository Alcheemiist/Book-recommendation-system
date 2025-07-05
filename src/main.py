import os
from dotenv import load_dotenv
from pipeline import run_data_pipeline
from services.recommendation_service import test_vector_store, recommend_books
from core.text_preprocessor import TextPreprocessor

load_dotenv()

def main():
    """Main function to run the book recommendation system"""
    
    print("📚 Book Recommendation System")
    print("=" * 50)
    
    # Check if vector store exists
    if os.path.exists("../book_index"):
        print("✅ Vector store found!")
        
        # Test vector store
        if test_vector_store():
            print("\n🎯 Testing recommendation service...\n")
            query = "machine learning and artificial intelligence"

            print(f"query : {query}")

            preprocessor = TextPreprocessor()
            query = preprocessor.preprocess_text(query)

            print(f"Preprocessed query: {query}")

            results = recommend_books(query, 3)
            print(f"\nTop 3 recommendations for '{query}':")
            print(results[['title', 'authors', 'similarity_score']].to_string(index=False))
        else:
            print("❌ Vector store test failed. Running pipeline...")
            run_data_pipeline()
    else:
        print("❌ Vector store not found. Running pipeline...")
        run_data_pipeline()

if __name__ == "__main__":
    main() 