import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing duplicates and missing values"""
    df = df.dropna(subset=["description", "title"])
    df = df.drop_duplicates(subset=["description", "title"])
    df["description"] = df["description"].str.replace(r'\s+', ' ', regex=True)
    df["description"] = df["description"].str.strip()
    df = df[df["description"] != ""]
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def embedding_data(df_books: pd.DataFrame, embeddings_model: OpenAIEmbeddings, use_preprocessed: bool = True) -> pd.DataFrame:
    """Generate embeddings for book descriptions"""
    df = df_books.copy()
    
    # Use preprocessed descriptions if available
    text_column = 'description_processed' if use_preprocessed and 'description_processed' in df.columns else 'description'

    # Generate embeddings
    print(f"Generating embeddings for {len(df)} books using column: {text_column}")
    texts = df[text_column].tolist()
    df['embedding'] = embeddings_model.embed_documents(texts)
    
    # Save embeddings to CSV
    df.to_csv("../data/books_embeddings.csv", index=False)
    print("Embeddings saved to ../data/books_embeddings.csv")
    return df

def create_and_save_vector_store(df_books: pd.DataFrame, embeddings_model: OpenAIEmbeddings, use_preprocessed: bool = True) -> str:
    """Create and save FAISS vector store"""
    df = df_books.copy()
    
    # Use preprocessed descriptions if available
    text_column = 'description_processed' if use_preprocessed and 'description_processed' in df.columns else 'description'
    
    print(f"Creating vector store with {len(df)} books using column: {text_column}")
    
    # Prepare texts and metadata
    texts = df[text_column].tolist()
    metadatas = []
    
    for _, row in df.iterrows():
        metadata = {
            "id": row["id"],
            "title": row["title"],
            "authors": row["authors"],
            "categories": row.get("categories", ""),
            "primary_category": row.get("primary_category", "General"),
            "search_strategy": row.get("search_strategy", ""),
            "retrieval_timestamp": row.get("retrieval_timestamp", "")
        }
        metadatas.append(metadata)
    
    # Create vector store
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings_model,
        metadatas=metadatas
    )
    
    # Save vector store
    vector_store_path = "../book_index/"
    vector_store.save_local(vector_store_path)
    print("Vector store saved to ", vector_store_path)
    return vector_store_path 