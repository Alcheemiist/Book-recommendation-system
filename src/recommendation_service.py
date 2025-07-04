import pandas as pd
from langchain_community.vectorstores import FAISS

def recommend_books(query, k=10):
    """
    Recommend books based on a query using a vector store Similarity search. AND return results as a Pandas DataFrame.
    """
    # Get similar documents
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
    print(results.to_string(index=False, columns=["title", "authors", "description","similarity_score"]))

if "__name__" == "__main__":
    vector_store = FAISS.load_local("./book_index")

    results = recommend_books("machine learning", k=5)
    visualize_results(results)