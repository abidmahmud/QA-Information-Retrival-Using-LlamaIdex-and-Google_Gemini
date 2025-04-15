from llama_index.core import VectorStoreIndex
from llama_index.core import Settings  # <-- Replace ServiceContext import
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter  # <-- Add node parser

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model,document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("Initializing embedding model and settings")
        
        # Initialize components
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)  # <-- Create node parser
        
        # Configure global settings
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.node_parser = node_parser  # <-- Set node parser in settings
        
        logging.info("Creating vector store")
        index = VectorStoreIndex.from_documents(document)  # <-- Remove service_context parameter
        index.storage_context.persist()
        
        logging.info("Creating query engine")
        query_engine = index.as_query_engine()
        return query_engine
        
    except Exception as e:
        raise customexception(e,sys)
