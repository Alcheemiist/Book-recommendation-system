import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing duplicates and missing values"""
    df = df.dropna(subset=["description"])
    df = df.drop_duplicates(subset=["description"])
    df = df.dropna(subset=["title"])
    df = df.drop_duplicates(subset=["title"])
    df["description"] = df["description"].str.replace(r'\s+', ' ', regex=True)
    df["description"] = df["description"].str.strip()
    df = df[df["description"] != ""]
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def embed_data(df_books: pd.DataFrame, 
               embeddings_model: OpenAIEmbeddings, 
               use_preprocessed: bool = True) -> pd.DataFrame:
    """Generate embeddings for book descriptions"""
    df = df_books.copy()
    
    # Use preprocessed descriptions if available
    text_column = 'description_processed' if use_preprocessed and 'description_processed' in df.columns else 'description'
    
    print(f"Generating embeddings for {len(df)} books using column: {text_column}")
    
    # Generate embeddings
    texts = df[text_column].tolist()
    embeddings = embeddings_model.embed_documents(texts)
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings
    
    # Save embeddings to CSV
    df.to_csv("../data/books_embeddings.csv", index=False)
    print(f"Embeddings saved to ../data/books_embeddings.csv")
    
    return df

def create_and_save_vector_store(df_books: pd.DataFrame, 
                                embeddings_model: OpenAIEmbeddings, 
                                use_preprocessed: bool = True) -> FAISS:
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
            "primary_category": row.get("primary_category", "General")
        }
        metadatas.append(metadata)
    
    # Create vector store
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings_model,
        metadatas=metadatas
    )
    
    # Save vector store
    vector_store.save_local("../book_index")
    print("Vector store saved to ../book_index/")
    
    return vector_store 