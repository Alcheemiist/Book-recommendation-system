import os
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
import requests
from tqdm import tqdm
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

class SystematicGoogleBooksLoader(BaseLoader):
    """Loader that fetches books per search value systematically"""
    
    def __init__(self):
        self.BASE_URL = "https://www.googleapis.com/books/v1/volumes"
        self.RESULTS_PER_PAGE = 10
        self.MIN_BOOKS_PER_VALUE = 5
        self.MIN_DESCRIPTION_LENGTH = 2048
        self.API_KEY = os.getenv("GOOGLE_API_KEY")
        
        self.SEARCH_STRATEGIES = [
            {
                "param": "subject", 
                "values": [ "Fiction", "Science", "History", "Biography", "Technology", "Art", "Philosophy", "Psychology", "Business", "Travel"]
            },
            {
                "param": "inauthor", 
                "values": ["Stephen King", "J.K. Rowling", "Yuval Noah Harari", "Michelle Obama", "Malcolm Gladwell", "J.R.R. Tolkien", "Agatha Christie", "Toni Morrison"]
            },
            {
                "param": "intitle", 
                "values": [ "Python", "Machine Learning", "World War", "Artificial Intelligence", "Space Exploration", "Climate Change", "Ancient History", "Mental Health"]
            }
        ]
        
        self.books_data = {}  # book_id -> book_data
        self.value_counts = {}  # (param, value) -> count

    def load(self) -> pd.DataFrame:
        """Fetch books systematically for all strategy values"""
        print(f"Fetching at least {self.MIN_BOOKS_PER_VALUE} books per search value...")
        
        # Initialize counters
        for strategy in self.SEARCH_STRATEGIES:
            for value in strategy["values"]:
                self.value_counts[(strategy["param"], value)] = 0
        
        # Process each search value systematically
        for strategy in tqdm(self.SEARCH_STRATEGIES, desc="Processing strategies"):
            param = strategy["param"]
            for value in tqdm(strategy["values"], desc=f"Processing {param} values"):
                self._fetch_for_value(param, value)
                time.sleep(1)
        
        print(f"\nTotal books collected: {len(self.books_data)}")
        return pd.DataFrame(self.books_data.values())

    def _fetch_for_value(self, param: str, value: str) -> None:
        """Fetch books for a specific parameter value until we have enough"""
        query = f"{param}:{value}"
        start_index = 0
        collected = 0
        retry_count = 0
        max_retries = 3
        
        while collected < self.MIN_BOOKS_PER_VALUE:
            try:
                response = requests.get(
                    self.BASE_URL,
                    params={
                        "q": query,
                        "maxResults": self.RESULTS_PER_PAGE,
                        "startIndex": start_index,
                        "key": self.API_KEY,
                        "printType": "books",
                        "langRestrict": "en"
                    },
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                
                if "items" not in data:
                    print(f"\nNo more results for '{query}'")
                    break
                
                new_books = self._process_items(data["items"], param, value)
                collected += new_books
                start_index += self.RESULTS_PER_PAGE
                retry_count = 0  # Reset retry count on success
                
                # Break if API returns fewer items than requested
                if len(data.get("items", [])) < self.RESULTS_PER_PAGE:
                    break
                    
                time.sleep(2)  # Increased delay between successful requests
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"\nMax retries exceeded for '{query}'. Skipping...")
                        break
                    wait_time = min(60 * (2 ** retry_count), 300)  # Exponential backoff, max 5 minutes
                    print(f"\nRate limited for '{query}'. Waiting {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\nHTTP Error for '{query}': {str(e)}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"\nAPI Error for '{query}': {str(e)}")
                time.sleep(10)
                continue
        
        print(f"\nCollected {collected} books for {param}:{value}")

    def _process_items(self, items: List[Dict], param: str, value: str) -> int:
        """Process API items and return count of new books added"""
        new_books = 0
        for item in items:
            volume_info = item.get("volumeInfo", {})
            book_id = item.get("id")
            
            # Skip if doesn't meet criteria
            if (not volume_info.get("description") or 
                len(volume_info["description"]) < self.MIN_DESCRIPTION_LENGTH or
                book_id in self.books_data):
                continue
            
            # Store book data
            self.books_data[book_id] = {
                "id": book_id,
                "title": volume_info.get("title", "Untitled"),
                "authors": ", ".join(volume_info.get("authors", ["Unknown"])),
                "description": volume_info.get("description", ""),
                "categories": ", ".join(volume_info.get("categories", [])),
                "primary_category": volume_info.get("categories", ["General"])[0] if volume_info.get("categories") else "General",
               
                "search_strategy": f"{param}:{value}",
                "retrieval_timestamp": datetime.now().isoformat()
            }
            self.value_counts[(param, value)] += 1
            new_books += 1
        
        return new_books

    def get_coverage_report(self) -> Dict:
        """Generate report showing books collected per search value"""
        report = {}
        for strategy in self.SEARCH_STRATEGIES:
            param = strategy["param"]
            report[param] = {}
            for value in strategy["values"]:
                report[param][value] = self.value_counts.get((param, value), 0)
        return report
    
    def save_to_csv(self, filename: str) -> None:
        """Save books data to CSV file"""
        df = pd.DataFrame(self.books_data.values())
        df.to_csv(filename, index=False)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean the data description by removing duplicates and missing values and whitespace"""
    
    df = df.dropna(subset=["description"])
    df = df.drop_duplicates(subset=["description"])
    df = df.dropna(subset=["title"])
    df = df.drop_duplicates(subset=["title"])
    df["description"] = df["description"].str.replace(r'\s+', ' ', regex=True)
    df["description"] = df["description"].str.strip()
    df = df[df["description"] != ""]
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")
    return df 

def load_data_from_google_books(loader) -> pd.DataFrame:
    """
    Load data from google books and return loader, dataframe, and report
    """
    df_books = loader.load()
    report = loader.get_coverage_report()

    print("\nSearch Strategy Coverage:")
    for param, values in report.items():
        print(f"\n{param.upper()}:")
        for value, count in values.items():
            print(f"  {value}: {count} books")

    print("\nSample document:")
    sample = df_books.iloc[0]
    print(f"ID: {sample['id']}\nTitle: {sample['title']}\nAuthors: {sample['authors']}\nDescription: {sample['description'][:100]}...")
    print(f"Categories: {sample['categories']}\nPrimary Category: {sample['primary_category']}\nSearch Strategy: {sample['search_strategy']}\nRetrieval Timestamp: {sample['retrieval_timestamp']}")
    return df_books

def embed_data(df_books):
    """
    Embed data using OpenAIEmbeddings and return embeddings model and dataframe with embeddings
    """
    embeddings_model = OpenAIEmbeddings( model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    # Batch processing for efficiency
    batch_size = 50
    embeddings = []
    for i in range(0, len(df_books), batch_size):
        batch = df_books["description"].iloc[i:i+batch_size].tolist()
        embeddings.extend(embeddings_model.embed_documents(batch))
        
    df_books["embedding"] = embeddings
    #df_books["embedding"] = df_books["embedding"].apply(lambda x: [float(i) for i in x])

    df_books.to_csv("../data/books_embeddings.csv")
    print(len(df_books))
    return df_books, embeddings_model

def create_and_save_vector_store(df_books, embeddings_model):
    """
    Create , save and return vector store
    """
    vector_store = FAISS.from_embeddings(
        text_embeddings=zip(df_books["description"], df_books["embedding"]),
        embedding=embeddings_model,
        metadatas=df_books.drop(columns=["description", "embedding"]).to_dict("records")
    )
    vector_store.save_local("../data/book_index")    
    return vector_store

def data_pipeline():
    """
    Load data from google books, clean data, embed data, and create and return vector store
    """
    # loader = SystematicGoogleBooksLoader()
    #df_books = load_data_from_google_books(loader)
    #df_books.to_csv("../data/books_1.csv", index=False)

    df_books = pd.read_csv("../data/books.csv")
    df_books = clean_data(df_books)

    # print("sample 0 : ")
    # print(f"ID: {df_books['id'].iloc[0]}")
    # print(f"Title: {df_books['title'].iloc[0]}")
    # print(f"Authors: {df_books['authors'].iloc[0]}")
    # print(f"Description: {df_books['description'].iloc[0]}")

    df_books, embeddings_model = embed_data(df_books)

    # print(f"embeddings type: {type(df_books['embedding'].iloc[0])}")
    # print(f"embeddings length: {len(df_books['embedding'].iloc[0])}")

    vector_store = create_and_save_vector_store(df_books, embeddings_model)
    return vector_store

if __name__ == "__main__":
    data_pipeline()
