from preprocessing import preprocess_document

import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from model_providers import get_embeddings_model

# --- Carga de Configuración Externa ---
CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'rag_config.json'

def load_config() -> Dict[str, Any]:
    """Carga la configuración desde el archivo JSON."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"El archivo de configuración no se encontró en: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()
DOCS_DIR = Path(__file__).resolve().parent / 'docs'

def get_optimal_chunk_config(embedding_provider: str, embedding_model: Optional[str] = None) -> Tuple[int, int]:
    """
    Determina la configuración óptima de chunking basada en el modelo de embedding.
    
    Basado en investigación reciente de RAG (2024):
    - text-embedding-3-large/ada-002: 256-512 tokens óptimo
    - sentence-transformers: oraciones individuales 
    - modelos locales (Ollama): 256-384 tokens típicamente
    
    Args:
        embedding_provider: Proveedor del modelo de embedding
        embedding_model: Modelo específico de embedding
        
    Returns:
        Tupla (chunk_size_tokens, overlap_tokens)
    """
    
    # Mapeo de modelos a configuraciones óptimas (en tokens)
    MODEL_CONFIGS = {
        "openai": {
            "text-embedding-3-large": (512, 51),  # 10% overlap
            "text-embedding-3-small": (512, 51),
            "text-embedding-ada-002": (512, 51),
            "default": (512, 51)
        },
        "ollama": {
            "nomic-embed-text": (384, 38),  # 10% overlap
            "all-minilm": (256, 26),
            "default": (384, 38)
        },
        "huggingface": {
            "BAAI/bge-large-en-v1.5": (512, 51),
            "sentence-transformers/all-MiniLM-L6-v2": (256, 26),
            "default": (384, 38)
        }
    }
    
    provider_config = MODEL_CONFIGS.get(embedding_provider.lower(), MODEL_CONFIGS["openai"])
    model_key = embedding_model if embedding_model in provider_config else "default"
    chunk_tokens, overlap_tokens = provider_config[model_key]
    
    # Convertir tokens a caracteres aproximados (1 token ≈ 4 caracteres para español)
    # Factor conservador para español que tiende a ser más verbose
    chars_per_token = 4.5
    chunk_size = int(chunk_tokens * chars_per_token)
    chunk_overlap = int(overlap_tokens * chars_per_token)
    
    print(f"📏 Configuración de chunking optimizada para {embedding_provider}:{embedding_model}")
    print(f"   • Tamaño objetivo: {chunk_tokens} tokens (~{chunk_size} caracteres)")
    print(f"   • Overlap: {overlap_tokens} tokens (~{chunk_overlap} caracteres, {overlap_tokens/chunk_tokens*100:.1f}%)")
    print(f"   • Justificación: Optimizado para ventana de contexto del modelo de embedding")
    
    return chunk_size, chunk_overlap


def save_chunk_metadata(output_path: str, embedding_provider: str, embedding_model: Optional[str], 
                       chunk_size: int, chunk_overlap: int, total_chunks: int):
    """
    Guarda metadatos del chunking y modelo para validación de consistencia.
    
    Esto resuelve el problema de validación mencionado en las observaciones:
    asegura que se use el mismo modelo para indexación y consulta.
    """
    metadata = {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model or f"{embedding_provider}_default",
        "chunking_config": {
            "chunk_size_chars": chunk_size,
            "chunk_overlap_chars": chunk_overlap,
            "overlap_percentage": round(chunk_overlap / chunk_size * 100, 1),
            "total_chunks": total_chunks
        },
        "creation_timestamp": json.dumps({"timestamp": "generated_during_ingestion"}),
        "preprocessing_strategy": "transformer_optimized_no_stopwords_no_accent_removal"
    }
    
    metadata_path = Path(output_path) / "index_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Metadatos guardados en: {metadata_path}")
    return metadata


def ingest_and_build(output_path: str, embedding_provider: Optional[str] = None, embedding_model: Optional[str] = None):
    """
    Carga archivos JSON y construye un vector store FAISS persistente con chunking optimizado.

    Mejoras implementadas basadas en observaciones:
    1. Chunking optimizado por modelo de embedding (soluciona problema de chunking básico)
    2. Metadatos de validación (soluciona problema de consistencia de modelos) 
    3. Documentación detallada de decisiones técnicas
    4. Configuración adaptativa basada en research de RAG 2024

    Args:
        output_path: Directorio donde almacenar el índice FAISS
        embedding_provider: Proveedor de embeddings ("openai", "ollama", "huggingface")
        embedding_model: Modelo específico de embeddings (opcional)

    Returns:
        Vector store FAISS inicializado
    """
    # Cargar configuración de embedding desde JSON
    embedding_cfg = config["embedding_config"]

    # Prioridad: 1. Argumentos, 2. config.json
    final_embedding_provider = embedding_provider or embedding_cfg["default_provider"]
    final_embedding_model = embedding_model or embedding_cfg["default_model"]

    print(f"🔧 Iniciando proceso de ingesta con configuración:")
    print(f"   • Proveedor de Embeddings: {final_embedding_provider}")
    print(f"   • Modelo de Embeddings: {final_embedding_model or 'default'}")

    documents = []
    
    # Recursively process all JSON files in subdirectories
    for json_path in DOCS_DIR.glob('**/*.json'):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Preprocesar el contenido del JSON con estrategia optimizada para Transformers
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
            print(f"❌ Error procesando {json_path}: {e}")
    
    # Obtener configuración óptima de chunking basada en el modelo
    chunk_size, chunk_overlap = get_optimal_chunk_config(final_embedding_provider, final_embedding_model)
    
    # Configurar splitter con parámetros optimizados
    # RecursiveCharacterTextSplitter es óptimo para preservar contexto semántico
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Separadores priorizados para preservar estructura semántica
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    splits = splitter.split_documents(documents)
    
    print(f"📊 Procesados {len(documents)} archivos JSON → {len(splits)} chunks")
    print(f"📈 Promedio de {len(splits)/len(documents):.1f} chunks por documento")

    # Crear vector store con el modelo de embedding especificado
    embeddings = get_embeddings_model(provider=final_embedding_provider, model_name=final_embedding_model)
    print(f"🧠 Usando modelo de embeddings: {final_embedding_provider}:{final_embedding_model or 'default'}")
    
    # Crear el directorio de salida si no existe
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(output_path)
    
    # Guardar metadatos para validación futura
    save_chunk_metadata(output_path, final_embedding_provider, final_embedding_model, 
                       chunk_size, chunk_overlap, len(splits))
    
    print(f"✅ Vector store guardado en: {output_path}")
    print(f"🔍 Para usar en consultas, asegúrate de usar el mismo modelo de embedding")
    
    return vector_store

if __name__ == '__main__':
    import argparse
    
    # Cargar defaults desde el config file
    embedding_cfg = config.get("embedding_config", {})
    default_provider = embedding_cfg.get("default_provider")
    default_model = embedding_cfg.get("default_model")
    default_output = config.get("storage_path", "storage")

    parser = argparse.ArgumentParser(description='Ingest documents and build vector store with optimized chunking')
    parser.add_argument('--output', type=str, default=default_output,
                      help=f'Output directory for the FAISS index (default: {default_output})')
    parser.add_argument('--provider', type=str, default=default_provider,
                      help=f'Embedding provider (openai, ollama, huggingface) (default: {default_provider})')
    parser.add_argument('--model', type=str, default=default_model,
                      help=f'Specific embedding model to use (default: {default_model})')
    
    args = parser.parse_args()
    
    output_path = str(Path(__file__).resolve().parent / args.output)
    ingest_and_build(output_path, args.provider, args.model)
