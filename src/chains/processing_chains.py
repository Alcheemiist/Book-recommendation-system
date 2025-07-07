from typing import List, Dict, Any
from pydantic import Field
from core.base_chain import MonitoredDataChain
from core.text_preprocessor import preprocessing_pipeline
from core.data_processor import embedding_data, create_and_save_vector_store
from core.data_processor import clean_data

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
        cleaned_data = clean_data(raw_data)
        print(f"Cleaned dataset shape: {cleaned_data.shape}")
        return {self.output_key: cleaned_data}
    
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
        preprocessor = preprocessing_pipeline(self.config)
        processed_data['description_processed'] = processed_data['description'].apply(preprocessor.
        preprocess_text)
        return {self.output_key: processed_data}
    
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
        """Generate embeddings for preprocessed descriptions"""
        preprocessed_data = inputs[self.input_key]
        embedded_data = embedding_data(preprocessed_data, self.embeddings_model, use_preprocessed=True)
        return {self.output_key: embedded_data}
       
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
        vector_store_path =  create_and_save_vector_store(embedded_data, self.embeddings_model)
        return {self.output_key: vector_store_path}