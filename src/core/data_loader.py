import os
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders.base import BaseLoader
import requests
from tqdm import tqdm
import pandas as pd

load_dotenv()

class GoogleBooksLoader(BaseLoader):
    """Loader that fetches books from Google Books API systematically"""
    
    def __init__(self):
        self.BASE_URL = "https://www.googleapis.com/books/v1/volumes"
        self.RESULTS_PER_PAGE = 30
        self.MIN_BOOKS_PER_VALUE = 30
        self.MIN_DESCRIPTION_LENGTH = 2048
        self.API_KEY = os.getenv("GOOGLE_API_KEY")
        
        self.SEARCH_STRATEGIES = [
            {
                "param": "subject", 
                "values": ["Fiction", "Science", "History", "Biography", "Technology", 
                          "Art", "Philosophy", "Psychology", "Business", "Travel"]
            },
            {
                "param": "inauthor", 
                "values": ["Stephen King", "J.K. Rowling", "Yuval Noah Harari", 
                          "Michelle Obama", "Malcolm Gladwell", "J.R.R. Tolkien", 
                          "Agatha Christie", "Toni Morrison"]
            },
            {
                "param": "intitle", 
                "values": ["Python", "Machine Learning", "World War", 
                          "Artificial Intelligence", "Space Exploration", 
                          "Climate Change", "Ancient History", "Mental Health"]
            }
        ]
        
        self.books_data = {}  # book_id -> book_data
        self.value_counts = {}  # (param, value) -> count

    def load(self) -> pd.DataFrame:
        """Fetch books systematically for all strategy values"""
        print(f"Fetching at least {self.MIN_BOOKS_PER_VALUE} books per search value...")
        
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
        result_df = pd.DataFrame(self.books_data.values())
        result_df.to_csv("../data/books.csv", index=False)
        return result_df

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
                retry_count = 0
                
                if len(data.get("items", [])) < self.RESULTS_PER_PAGE:
                    break
                    
                time.sleep(2)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"\nMax retries exceeded for '{query}'. Skipping...")
                        break
                    wait_time = min(60 * (2 ** retry_count), 300)
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