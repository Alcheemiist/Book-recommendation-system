import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import pandas as pd
import unicodedata
from typing import List, Dict

# Download required NLTK data
def _ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    required_packages = [
        'punkt', 'stopwords', 'wordnet', 
        'averaged_perceptron_tagger'
    ]
    
    for package in required_packages:
        try:
            if 'punkt' in package:
                nltk.data.find('tokenizers/punkt')
            elif package in ['stopwords', 'wordnet']:
                nltk.data.find(f'corpora/{package}')
            else:
                nltk.data.find(f'taggers/{package}')
        except LookupError:
            nltk.download(package)

_ensure_nltk_data()

class TextPreprocessor:
    """Advanced text preprocessing for book descriptions"""
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 use_lemmatization: bool = True,
                 remove_punctuation: bool = True,
                 normalize_case: bool = True,
                 remove_numbers: bool = False,
                 min_word_length: int = 2):
        
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.remove_punctuation = remove_punctuation
        self.normalize_case = normalize_case
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Book-specific stop words
        self.book_stop_words = {
            'book', 'books', 'read', 'reading', 'author', 'authors', 'publisher',
            'published', 'publication', 'edition', 'volume', 'series', 'chapter',
            'pages', 'page', 'hardcover', 'paperback', 'ebook', 'digital'
        }
        self.stop_words.update(self.book_stop_words)
    
    def normalize_text(self, text: str) -> str:
        """Basic text normalization"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        text = unicodedata.normalize('NFKC', text)
        
        if self.normalize_case:
            text = text.lower()
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens based on length and other criteria"""
        filtered_tokens = []
        
        for token in tokens:
            if len(token) < self.min_word_length:
                continue
            
            if self.remove_numbers and token.isdigit():
                continue
            
            if token.lower() in self.stop_words:
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def get_wordnet_pos(self, tag: str) -> str:
        """Convert POS tag to WordNet POS tag"""
        tag_dict = {
            "J": nltk.corpus.wordnet.ADJ,
            "N": nltk.corpus.wordnet.NOUN,
            "V": nltk.corpus.wordnet.VERB,
            "R": nltk.corpus.wordnet.ADV
        }
        return tag_dict.get(tag[0], nltk.corpus.wordnet.NOUN)
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens with POS tagging"""
        if not self.lemmatizer:
            return tokens
        
        pos_tags = pos_tag(tokens)
        lemmatized_tokens = []
        
        for token, tag in pos_tags:
            wn_pos = self.get_wordnet_pos(tag)
            lemmatized = self.lemmatizer.lemmatize(token, wn_pos)
            lemmatized_tokens.append(lemmatized)
        
        return lemmatized_tokens
    
    def preprocess_text(self, text: str) -> str:
        """Complete text preprocessing pipeline"""
        # Normalize text
        text = self.normalize_text(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Tokenize
        tokens = self.tokenize_text(text)
        
        # Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Lemmatize 
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join back to text
        return ' '.join(tokens)

def preprocessing_pipeline(config: Dict = None) -> TextPreprocessor:
    """Create a text preprocessing pipeline with given configuration"""
    default_config = {
        'remove_stopwords': True,
        'use_lemmatization': True,
        'remove_punctuation': True,
        'normalize_case': True,
        'remove_numbers': False,
        'min_word_length': 2
    }
    
    if config:
        default_config.update(config)
    
    return TextPreprocessor(**default_config)

if __name__ == "__main__":
    # Example usage
    preprocessor = preprocessing_pipeline()
    sample_text = "The quick brown fox jumps over the lazy dog. 12345, the pursuit of hapiness!"
    processed_text = preprocessor.preprocess_text(sample_text)
    print("query : ", sample_text)
    print("Processed Text:", processed_text)