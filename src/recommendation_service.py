import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

def recommend_books(query, k):
    """
    Recommend books based on a query using a vector store Similarity search. AND return results as a Pandas DataFrame.
    """
    embeddings_model = OpenAIEmbeddings( model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = FAISS.load_local("../book_index", embeddings_model, allow_dangerous_deserialization=True)
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
            "similarity_score": score
        })
    return pd.DataFrame(results)

def visualize_results(results):
    """
    Visualize the results in a simple text format.
    """
    print("\nRecommended Books:")
    for index, row in results.iterrows():
        print(f"\nID: {row['id']}")
        print(f"Similarity Score: {row['similarity_score']:.4f}")
        print(f"Title: {row['title']}")
        print(f"Authors: {row['authors']}")
        print(f"\nDescription:\n{row['description'][100:]}\n")
        print("-" * 80)

if __name__ == "__main__":
    query = "Discover how ancient civilizations molded us into who we are today Explore the highly developed Aztec Empire, their language, agriculture, military tradition, astronomy, rituals, and art."
    results = recommend_books(query, 5)
    visualize_results(results)