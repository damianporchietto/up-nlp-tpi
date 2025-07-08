import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from model_providers import get_llm_model, get_embeddings_model
from ingest import ingest_and_build

# --- Carga de Configuración Externa ---
CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'rag_config.json'

def load_config() -> Dict[str, Any]:
    """Carga la configuración desde el archivo JSON."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"El archivo de configuración no se encontró en: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()

# --- Constantes y Plantillas ---
PROMPT_TEMPLATE = config["prompt_template"]
STORAGE_PATH = Path(__file__).resolve().parent / config["storage_path"]


class ModelValidationError(Exception):
    """Excepción para errores de validación de modelos"""
    pass


def validate_embedding_consistency(storage_path: Path, embedding_provider: str, embedding_model: Optional[str]) -> Dict[str, Any]:
    """
    Valida que el modelo de embedding usado para consultas sea consistente con el usado para indexación.
    
    Esta función resuelve el problema crítico mencionado en las observaciones:
    "asegurar que se use el mismo modelo de embedding para la indexación de los documentos y para la consulta del usuario"
    
    Args:
        storage_path: Ruta al índice FAISS
        embedding_provider: Proveedor del modelo de embedding para consultas
        embedding_model: Modelo específico de embedding para consultas
        
    Returns:
        Metadatos del índice si la validación es exitosa
        
    Raises:
        ModelValidationError: Si hay inconsistencia entre modelos
    """
    metadata_path = storage_path / "index_metadata.json"
    
    if not metadata_path.exists():
        raise ModelValidationError(
            f"❌ No se encontraron metadatos del índice en {metadata_path}. "
            "Esto puede indicar que el índice fue creado con una versión anterior. "
            "Recrea el índice ejecutando: python ingest.py"
        )
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        index_metadata = json.load(f)
    
    index_provider = index_metadata.get("embedding_provider")
    index_model = index_metadata.get("embedding_model")
    
    # Normalizar modelo actual para comparación
    current_model = embedding_model or f"{embedding_provider}_default"
    
    # Validar consistencia
    if index_provider != embedding_provider:
        raise ModelValidationError(
            f"❌ Inconsistencia de proveedor de embedding:\n"
            f"   • Índice creado con: {index_provider}\n"
            f"   • Consulta usando: {embedding_provider}\n"
            f"Solución: Usa --embedding-provider {index_provider} o recrea el índice."
        )
    
    if index_model != current_model:
        raise ModelValidationError(
            f"❌ Inconsistencia de modelo de embedding:\n"
            f"   • Índice creado con: {index_model}\n"
            f"   • Consulta usando: {current_model}\n"
            f"Solución: Usa --embedding-model {index_model.replace(f'{index_provider}_', '') if index_provider in index_model else index_model} o recrea el índice."
        )
    
    print(f"✅ Validación de modelo exitosa:")
    print(f"   • Proveedor: {index_provider}")
    print(f"   • Modelo: {index_model}")
    print(f"   • Estrategia chunking: {index_metadata.get('chunking_config', {}).get('chunk_size_chars', 'N/A')} chars")
    print(f"   • Total chunks: {index_metadata.get('chunking_config', {}).get('total_chunks', 'N/A')}")
    
    return index_metadata


class RAGPipeline:
    def __init__(self, 
                 llm_provider: str = "openai", 
                 llm_model: Optional[str] = None,
                 embedding_provider: str = "openai", 
                 embedding_model: Optional[str] = None,
                 temperature: float = 0):
        """
        Inicializa el pipeline RAG con validación de consistencia de modelos.
        
        Mejoras implementadas basadas en observaciones:
        1. Validación obligatoria de consistencia de modelos de embedding
        2. Documentación clara de qué modelo se usa para qué
        3. Manejo robusto de errores de configuración
        4. Metadatos detallados para troubleshooting
        
        Args:
            llm_provider: Proveedor para el LLM ("openai", "ollama", "huggingface") - usado para GENERACIÓN
            llm_model: Modelo específico de LLM - usado para GENERACIÓN
            embedding_provider: Proveedor para embeddings ("openai", "ollama", "huggingface") - usado para BÚSQUEDA
            embedding_model: Modelo específico de embeddings - usado para BÚSQUEDA
            temperature: Temperatura para generación de texto
        """
        
        # Cargar configuración
        llm_cfg = config["llm_config"]
        embedding_cfg = config["embedding_config"]
        retriever_cfg = config["retriever_config"]
        
        # Sobrescribir con argumentos si se proporcionan
        self.llm_provider = llm_provider or llm_cfg["default_provider"]
        self.llm_model = llm_model or llm_cfg["default_model"]
        self.embedding_provider = embedding_provider or embedding_cfg["default_provider"]
        self.embedding_model = embedding_model or embedding_cfg["default_model"]
        self.temperature = temperature if temperature != 0 else llm_cfg["temperature"]

        print(f"🔧 Inicializando RAG Pipeline:")
        print(f"   • LLM (generación): {self.llm_provider}:{self.llm_model or 'default'}")
        print(f"   • Embeddings (búsqueda): {self.embedding_provider}:{self.embedding_model or 'default'}")
        
        # Inicializar modelo de embeddings
        self.embeddings = get_embeddings_model(provider=self.embedding_provider, model_name=self.embedding_model)
        
        # Validar o construir el vector store
        if STORAGE_PATH.exists() and any(STORAGE_PATH.iterdir()):
            try:
                # VALIDACIÓN CRÍTICA: Verificar consistencia de modelos
                self.index_metadata = validate_embedding_consistency(
                    STORAGE_PATH, self.embedding_provider, self.embedding_model
                )
                
                self.vector_store = FAISS.load_local(
                    str(STORAGE_PATH), 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                print(f"📚 Índice cargado desde: {STORAGE_PATH}")
                
            except ModelValidationError as e:
                print(f"\n{e}")
                print(f"\n🔄 Recomendación: Recrea el índice con los modelos correctos:")
                print(f"python ingest.py --provider {self.embedding_provider} --model {self.embedding_model or 'default'}")
                raise
                
        else:
            print(f"🏗️  Construyendo nuevo índice vectorial...")
            # Construir y persistir el vectorstore
            self.vector_store = ingest_and_build(
                str(STORAGE_PATH), 
                embedding_provider=self.embedding_provider, 
                embedding_model=self.embedding_model
            )
            
            # Cargar metadatos recién creados
            self.index_metadata = validate_embedding_consistency(
                STORAGE_PATH, self.embedding_provider, self.embedding_model
            )

        # Crear retriever con configuración optimizada desde JSON
        self.retriever = self.vector_store.as_retriever(
            search_type=retriever_cfg["search_type"],
            search_kwargs={"k": retriever_cfg["k"]}
        )
        
        # Inicializar LLM para generación
        self.llm = get_llm_model(provider=self.llm_provider, model_name=self.llm_model, temperature=self.temperature)
        
        # Crear prompt desde template
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Configurar cadena RAG usando LCEL (LangChain Expression Language)
        # Más estable que create_retrieval_chain
        self.chain = (
            {"query": RunnablePassthrough(), "context": self.retriever}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print(f"🚀 Pipeline RAG inicializado correctamente")

    def get_system_info(self) -> Dict[str, Any]:
        """
        Retorna información detallada del sistema para debugging y documentación.
        Resuelve el problema de especificación de modelos mencionado en las observaciones.
        """
        return {
            "embedding_model_info": {
                "provider": self.index_metadata.get("embedding_provider"),
                "model": self.index_metadata.get("embedding_model"), 
                "usage": "Búsqueda de similaridad en vector store"
            },
            "llm_model_info": {
                "provider": self.llm_provider,
                "model": self.llm_model or "default",
                "usage": "Generación de respuestas basadas en contexto"
            },
            "chunking_strategy": self.index_metadata.get("chunking_config"),
            "preprocessing_strategy": self.index_metadata.get("preprocessing_strategy"),
            "total_chunks": self.index_metadata.get("chunking_config", {}).get("total_chunks")
        }

    def __call__(self, question: str) -> Dict[str, Any]:
        try:
            # Ejecutar la cadena RAG
            answer = self.chain.invoke(question)
            
            # Obtener documentos del retriever directamente
            retrieved_docs = self.retriever.get_relevant_documents(question)
            
            # Formatear salida para consistencia con app.py
            return {
                "result": answer,
                "source_documents": retrieved_docs,
                "system_info": self.get_system_info()
            }
        except Exception as e:
            # Fallback a un enfoque más simple si la cadena falla
            print(f"⚠️ Error en cadena RAG: {str(e)}. Usando enfoque fallback.")
            
            # Fallback simple usando RetrievalQA
            simple_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True
            )
            
            result = simple_chain({"query": question})
            return {
                "result": result.get("result", ""),
                "source_documents": result.get("source_documents", []),
                "system_info": self.get_system_info(),
                "fallback_used": True
            }

# Convenience loader
_pipeline: RAGPipeline | None = None

def load_rag_chain(llm_provider: Optional[str] = None,
                  llm_model: Optional[str] = None,
                  embedding_provider: Optional[str] = None,
                  embedding_model: Optional[str] = None) -> RAGPipeline:
    """
    Carga o inicializa el pipeline RAG con modelos especificados.
    
    IMPORTANTE: Esta función incluye validación automática de consistencia de modelos.
    
    Args:
        llm_provider: Proveedor para el LLM (generación de respuestas)
        llm_model: Modelo específico de LLM
        embedding_provider: Proveedor para embeddings (búsqueda de similaridad)
        embedding_model: Modelo específico de embeddings
        
    Returns:
        Pipeline RAG inicializado y validado
        
    Raises:
        ModelValidationError: Si hay inconsistencia entre modelos de indexación y consulta
    """
    global _pipeline
    
    # Cargar configuración por defecto
    llm_cfg = config["llm_config"]
    embedding_cfg = config["embedding_config"]

    # Usar argumentos si se proporcionan, si no, usar config.json, si no, usar variables de entorno
    final_llm_provider = llm_provider or os.getenv("LLM_PROVIDER", llm_cfg["default_provider"])
    final_llm_model = llm_model or os.getenv("LLM_MODEL", llm_cfg["default_model"])
    final_embedding_provider = embedding_provider or os.getenv("EMBEDDING_PROVIDER", embedding_cfg["default_provider"])
    final_embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", embedding_cfg["default_model"])

    # Si los modelos cambian, recrear el pipeline
    if _pipeline is not None:
        if (os.getenv("LLM_PROVIDER") != final_llm_provider or 
            os.getenv("LLM_MODEL") != final_llm_model or
            os.getenv("EMBEDDING_PROVIDER") != final_embedding_provider or
            os.getenv("EMBEDDING_MODEL") != final_embedding_model):
            _pipeline = None
    
    if _pipeline is None:
        # Guardar configuración actual
        os.environ["LLM_PROVIDER"] = final_llm_provider
        os.environ["EMBEDDING_PROVIDER"] = final_embedding_provider
        if final_llm_model:
            os.environ["LLM_MODEL"] = final_llm_model
        if final_embedding_model:
            os.environ["EMBEDDING_MODEL"] = final_embedding_model
            
        _pipeline = RAGPipeline(
            llm_provider=final_llm_provider,
            llm_model=final_llm_model,
            embedding_provider=final_embedding_provider,
            embedding_model=final_embedding_model
        )
        
    return _pipeline