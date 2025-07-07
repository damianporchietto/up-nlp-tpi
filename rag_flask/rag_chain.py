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

STORAGE_PATH = Path(__file__).resolve().parent / 'storage'

# Custom prompt template for government documents
PROMPT_TEMPLATE = """
Eres un asistente especializado en trámites gubernamentales de la Provincia de Córdoba, Argentina.
Tienes acceso a información sobre procedimientos administrativos, requisitos y servicios.

Responde a la pregunta basándote únicamente en el siguiente contexto:

Context: {context}

Pregunta: {query}

Si la respuesta no está en el contexto proporcionado, indica que no tienes esa información y que el usuario 
debe consultar directamente con la oficina gubernamental correspondiente. No inventes información.

Tu respuesta debe ser clara, concisa y fácil de entender. Si hay requisitos o procedimientos específicos, 
enuméralos de manera organizada.

Respuesta:
"""


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
        
        print(f"🔧 Inicializando RAG Pipeline:")
        print(f"   • LLM (generación): {llm_provider}:{llm_model or 'default'}")
        print(f"   • Embeddings (búsqueda): {embedding_provider}:{embedding_model or 'default'}")
        
        # Inicializar modelo de embeddings
        self.embeddings = get_embeddings_model(provider=embedding_provider, model_name=embedding_model)
        
        # Validar o construir el vector store
        if STORAGE_PATH.exists() and any(STORAGE_PATH.iterdir()):
            try:
                # VALIDACIÓN CRÍTICA: Verificar consistencia de modelos
                self.index_metadata = validate_embedding_consistency(
                    STORAGE_PATH, embedding_provider, embedding_model
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
                print(f"python ingest.py --provider {embedding_provider} --model {embedding_model or 'default'}")
                raise
                
        else:
            print(f"🏗️  Construyendo nuevo índice vectorial...")
            # Construir y persistir el vectorstore
            self.vector_store = ingest_and_build(
                str(STORAGE_PATH), 
                embedding_provider=embedding_provider, 
                embedding_model=embedding_model
            )
            
            # Cargar metadatos recién creados
            self.index_metadata = validate_embedding_consistency(
                STORAGE_PATH, embedding_provider, embedding_model
            )

        # Crear retriever con configuración optimizada
        # k=4 es un buen balance para contexto sin saturar el LLM
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Inicializar LLM para generación
        self.llm = get_llm_model(provider=llm_provider, model_name=llm_model, temperature=temperature)
        
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
                "provider": os.getenv("LLM_PROVIDER", "unknown"),
                "model": os.getenv("LLM_MODEL", "unknown"),
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

def load_rag_chain(llm_provider: str = "openai", 
                  llm_model: Optional[str] = None,
                  embedding_provider: str = "openai", 
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
    
    # Si los modelos cambian, recrear el pipeline
    if _pipeline is not None:
        if (os.getenv("LLM_PROVIDER") != llm_provider or 
            os.getenv("EMBEDDING_PROVIDER") != embedding_provider):
            _pipeline = None
    
    if _pipeline is None:
        # Guardar configuración actual
        os.environ["LLM_PROVIDER"] = llm_provider
        os.environ["EMBEDDING_PROVIDER"] = embedding_provider
        if llm_model:
            os.environ["LLM_MODEL"] = llm_model
        if embedding_model:
            os.environ["EMBEDDING_MODEL"] = embedding_model
            
        _pipeline = RAGPipeline(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
        
    return _pipeline