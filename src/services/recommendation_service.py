import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

book_index =  "../book_index"

def recommend_books(query: str, k: int = 5) -> pd.DataFrame:
    """
    Recommend books based on a query using vector store similarity search.
    
    Args:
        query: The search query
        k: Number of recommendations to return
        
    Returns:
        DataFrame with recommended books and similarity scores
    """
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vector_store = FAISS.load_local(
        book_index, 
        embeddings_model, 
        allow_dangerous_deserialization=True
    )
    
    docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    
    # Convert to Pandas DataFrame
    results = []
    for doc, score in docs:
        meta = doc.metadata
        results.append({
            "id": meta["id"],
            "title": meta["title"],
            "authors": meta["authors"],
            "description": doc.page_content,
            "similarity_score": score,
            "categories": meta.get("categories", ""),
            "primary_category": meta.get("primary_category", "General"),
            "search_strategy": meta.get("search_strategy", ""),
            "retrieval_timestamp": meta.get("retrieval_timestamp", "")
        })
    
    return pd.DataFrame(results)

def test_vector_store() -> bool:
    """
    Test if the vector store is working properly.
    
    Returns:
        True if vector store is working, False otherwise
    """
    try:
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        vector_store = FAISS.load_local(
            book_index, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        
        # Test a simple query
        test_query = "test"
        docs = vector_store.similarity_search_with_relevance_scores(test_query, k=1)
        
        if docs:
            print("✅ Vector store is working properly!")
            print(f"Found {len(docs)} test results")
            return True
        else:
            print("❌ Vector store returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Error testing vector store: {str(e)}")
        return False

if __name__ == "__main__":
    # example usage of the recommendation function
    query = "Discover how ancient civilizations molded us into who we are today"
    results = recommend_books(query, 5)

    print("Recommended Books:")
    for _, row in results.iterrows():
        print(f"{_} - Title : {row['title']}, Authors: {row['authors']}, Similarity Score: {row['similarity_score']:.4f}")