from langchain.chains.base import Chain
from datetime import datetime
import time
from typing import Dict, Any
from pydantic import Field

class MonitoredDataChain(Chain):
    """Base chain with built-in monitoring and logging"""
    
    chain_name: str = Field(description="Name of the chain for monitoring")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Execution statistics")
    
    def __init__(self, chain_name: str, **kwargs):
        super().__init__(chain_name=chain_name, **kwargs)
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        result = self.execute_chain(inputs)
        
        end_time = time.time()
        self.stats = {
            'execution_time': end_time - start_time,
            'input_size': len(inputs),
            'output_size': len(result),
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_execution()
        return result
    
    def execute_chain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def log_execution(self):
        """Log chain execution statistics"""
        print(f"Chain '{self.chain_name}' completed:")
        print(f"  Execution time: {self.stats['execution_time']:.2f}s")
        print(f"  Input size: {self.stats['input_size']}")
        print(f"  Output size: {self.stats['output_size']}") 