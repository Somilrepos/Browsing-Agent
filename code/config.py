import os
import yaml
from getpass import getpass

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)    
    
    def setup_api_keys(self):
        """
        Set up API keys either from environment or user input
        """
        
        google_api_key = os.environ.get("GOOGLE_API_KEY") or self.get("api_keys", "GOOGLE_API_KEY") 
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        if not google_api_key:
            raise KeyError        
    
        return google_api_key
    
    
    def setup_langchain_tracing(self):
        """
        Set up LangChain tracing
        """
        
        if self.config['langchain']['tracing']['enabled']:
            os.environ["LANGSMITH_TRACING_V2"] = str(self.config['langchain']['tracing']['enabled']).lower() 
            os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY") or self.get("api_keys", "LANGCHAIN_API_KEY")
            os.environ["LANGSMITH_PROJECT"] = self.get("langchain", "project")
    
    
    def get(self, section, key=None):
        if key is None:
            return self.config.get(section, {})
        
        section_data = self.config.get(section, {})
        return section_data.get(key)

config = Config()
