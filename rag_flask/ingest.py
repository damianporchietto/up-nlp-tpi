from preprocessing import preprocess_document

import os
import json
from pathlib import Path
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from model_providers import get_embeddings_model

DOCS_DIR = Path(__file__).resolve().parent / 'docs'

def ingest_and_build(output_path: str, embedding_provider: str = "openai", embedding_model: Optional[str] = None):
    """Load JSON files under docs/ and build a persistent FAISS vector store.

    Args:
        output_path: Where to store the FAISS index directory
        embedding_provider: Provider for embeddings ("openai", "ollama", "huggingface")
        embedding_model: Optional specific model to use for embeddings

    Returns:
        Initialized FAISS vector store
    """
    documents = []
    
    # Recursively process all JSON files in subdirectories
    for json_path in DOCS_DIR.glob('**/*.json'):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            """    
            # Extract content from JSON structure
            content_parts = []
            content_parts.append(f"Title: {data.get('title', 'No title')}")
            content_parts.append(f"Description: {data.get('description', 'No description')}")
            
            # Process requirements if they exist
            if 'requirements' in data:
                for req in data['requirements']:
                    if 'title' in req and 'content' in req:
                        content_parts.append(f"{req['title']}: {req['content']}")
            
            # Create document with content and metadata
            content = "\n\n".join(content_parts)
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(json_path),
                        "title": data.get("title", ""),
                        "url": data.get("url", "")
                    }
                )
            )
            """
            # Preprocesar el contenido del JSON
            preprocessed = preprocess_document(data)
            
            # Crear el documento con el texto procesado
            documents.append(
                Document(
                    page_content=preprocessed["content"],
                    metadata={
                        "source": str(json_path),
                        "title": preprocessed["title"],
                        "url": preprocessed["url"]
                    }
                )
            )

        except Exception as e:
            print(f"Error processing {json_path}: {e}")
    
    # Split the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    
    print(f"Processed {len(documents)} JSON files into {len(splits)} chunks")

    # Create vector store with the specified embedding model
    embeddings = get_embeddings_model(provider=embedding_provider, model_name=embedding_model)
    print(f"Using embeddings model from provider: {embedding_provider}")
    
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(output_path)
    print(f"Vector store saved to {output_path}")
    return vector_store

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest documents and build vector store')
    parser.add_argument('--output', type=str, default='storage', 
                      help='Output directory for the FAISS index')
    parser.add_argument('--provider', type=str, default='openai',
                      help='Embedding provider (openai, ollama, huggingface)')
    parser.add_argument('--model', type=str, default=None,
                      help='Specific embedding model to use')
    
    args = parser.parse_args()
    
    output_path = str(Path(__file__).resolve().parent / args.output)
    ingest_and_build(output_path, args.provider, args.model)
