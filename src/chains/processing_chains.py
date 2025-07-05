from typing import List, Dict, Any
import re
import pandas as pd
from pydantic import Field
from core.base_chain import MonitoredDataChain
from core.text_preprocessor import create_preprocessing_pipeline
from core.data_processor import embed_data, create_and_save_vector_store

class DataCleaningChain(MonitoredDataChain):
    """Chain for automated data cleaning"""
    
    input_key: str = Field(default="raw_data", description="Input key for the chain")
    output_key: str = Field(default="cleaned_data", description="Output key for the chain")
    
    def __init__(self, **kwargs):
        super().__init__(chain_name="DataCleaning", **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def execute_chain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raw_data = inputs[self.input_key]
        cleaned_data = self.clean_data(raw_data)
        return {self.output_key: cleaned_data}
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Automated data cleaning"""
        df = data.copy()
        df = df.drop_duplicates()
        df['description'] = df['description'].apply(self.clean_description)
        df = df.dropna(subset=['title', 'description'])
        return df
    
    def clean_description(self, text: str) -> str:
        """Clean book description text"""
        if pd.isna(text):
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        return text

class TextPreprocessingChain(MonitoredDataChain):
    """Chain for text preprocessing"""
    
    config: Dict[str, Any] = Field(description="Configuration for text preprocessing")
    input_key: str = Field(default="cleaned_data", description="Input key for the chain")
    output_key: str = Field(default="preprocessed_data", description="Output key for the chain")
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(chain_name="TextPreprocessing", config=config, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def execute_chain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        processed_data = inputs[self.input_key]
        preprocessed_data = self.preprocess_texts(processed_data)
        return {self.output_key: preprocessed_data}
    
    def preprocess_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text preprocessing to descriptions"""
        df = df.copy()
        preprocessor = create_preprocessing_pipeline(self.config)
        df['description_processed'] = df['description'].apply(preprocessor.preprocess_text)
        return df

class EmbeddingGenerationChain(MonitoredDataChain):
    """Chain for embedding generation"""
    
    embeddings_model: Any = Field(description="Embeddings model to use")
    input_key: str = Field(default="preprocessed_data", description="Input key for the chain")
    output_key: str = Field(default="embedded_data", description="Output key for the chain")
    
    def __init__(self, embeddings_model, **kwargs):
        super().__init__(chain_name="EmbeddingGeneration", embeddings_model=embeddings_model, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def execute_chain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        preprocessed_data = inputs[self.input_key]
        embedded_data = self.generate_embeddings(preprocessed_data)
        return {self.output_key: embedded_data}
    
    def generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for processed descriptions"""
        return embed_data(df, self.embeddings_model, use_preprocessed=True)

class VectorStoreCreationChain(MonitoredDataChain):
    """Chain for vector store creation"""
    
    embeddings_model: Any = Field(description="Embeddings model to use")
    input_key: str = Field(default="embedded_data", description="Input key for the chain")
    output_key: str = Field(default="vector_store_path", description="Output key for the chain")
    
    def __init__(self, embeddings_model, **kwargs):
        super().__init__(chain_name="VectorStoreCreation", embeddings_model=embeddings_model, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def execute_chain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embedded_data = inputs[self.input_key]
        vector_store_path = self.create_vector_store(embedded_data)
        return {self.output_key: vector_store_path}
    
    def create_vector_store(self, df: pd.DataFrame) -> str:
        """Create FAISS vector store"""
        create_and_save_vector_store(df, self.embeddings_model)
        return "../book_index/" 