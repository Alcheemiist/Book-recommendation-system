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

def embedding_data(df_books: pd.DataFrame, embeddings_model: OpenAIEmbeddings, use_preprocessed: bool = True, batch_size: int = 300) -> pd.DataFrame:
    """Generate embeddings for book descriptions in batches of 300 rows"""
    df = df_books.copy()
    text_column = 'description_processed' if use_preprocessed and 'description_processed' in df.columns else 'description'
    print(f"Generating embeddings for {len(df)} books using column: {text_column}")
    texts = df[text_column].tolist()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    df['embedding'] = all_embeddings
    df.to_csv("../data/books_embeddings.csv", index=False)
    print("Embeddings saved to ../data/books_embeddings.csv")
    return df

def create_and_save_vector_store(df_books: pd.DataFrame, embeddings_model: OpenAIEmbeddings, use_preprocessed: bool = True) -> str:
    """Create and save FAISS vector store using precomputed embeddings"""
    df = df_books.copy()
    text_column = 'description_processed' if use_preprocessed and 'description_processed' in df.columns else 'description'

    print(f"Creating vector store with {len(df)} books using column: {text_column}")
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
    # Use embeddings from the DataFrame
    embeddings = df['embedding'].tolist()
    
    # Prepare text_embeddings as list of (text, embedding) tuples
    text_embeddings = list(zip(texts, embeddings))
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings_model,
        metadatas=metadatas
    )
    vector_store_path = "../book_index/"
    vector_store.save_local(vector_store_path)
    print("Vector store saved to ", vector_store_path)
    return vector_store_path 