import os
import argparse
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
import traceback
from flask_cors import CORS
import json
from pathlib import Path

from rag_chain import load_rag_chain, ModelValidationError
from model_providers import list_available_providers

load_dotenv()  # Loads OPENAI_API_KEY and other vars from .env if present

app = Flask(__name__)
CORS(app)

# Global RAG pipeline
rag_pipeline = None

# --- Variables de configuración (ahora manejadas por los argumentos de la app) ---
llm_provider = None
llm_model = None
embedding_provider = None
embedding_model = None

# Simple HTML documentation for the API - Note the double curly braces for CSS
API_DOCS = """
<!DOCTYPE html>
<html>
<head>
    <title>Asistente de Trámites de Córdoba - API</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        code {{ background: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background: #f8f8f8; padding: 10px; border-radius: 5px; overflow: auto; }}
        .endpoint {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .tester {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .tester textarea {{ width: 100%; height: 80px; padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 3px; }}
        .tester button {{ background-color: #2c3e50; color: white; border: none; padding: 10px 15px; border-radius: 3px; cursor: pointer; }}
        .tester button:hover {{ background-color: #1a2530; }}
        #result {{ margin-top: 20px; white-space: pre-wrap; }}
        .loading {{ display: none; margin-left: 10px; color: #666; }}
    </style>
</head>
<body>
    <h1>Asistente de Trámites de Córdoba - API</h1>
    <p>API para consultas sobre trámites y servicios gubernamentales de la Provincia de Córdoba.</p>
    
    <!-- Interactive test form -->
    <div class="tester">
        <h2>Probar API</h2>
        <p>Escribe una consulta y haz clic en "Enviar" para probar el asistente:</p>
        <textarea id="query" placeholder="Ejemplo: ¿Qué necesito para obtener un certificado de antecedentes?"></textarea>
        <div>
            <button onclick="testApi()">Enviar Consulta</button>
            <span id="loading" class="loading">Procesando consulta...</span>
        </div>
        <div id="result"></div>
    </div>
    
    <div class="endpoint">
        <h2>Consultar sobre trámites</h2>
        <p><strong>Endpoint:</strong> <code>POST /ask</code></p>
        <p><strong>Descripción:</strong> Consulta información sobre trámites y servicios gubernamentales.</p>
        <p><strong>Ejemplo de solicitud:</strong></p>
        <pre>
{{
    "message": "¿Qué necesito para obtener un certificado de antecedentes?"
}}
        </pre>
        <p><strong>Ejemplo de respuesta:</strong></p>
        <pre>
{{
    "answer": "Para obtener un certificado de antecedentes en la Provincia de Córdoba necesitas...",
    "sources": [
        {{
            "source": "/path/to/document.json",
            "snippet": "Fragmento del documento fuente..."
        }}
    ]
}}
        </pre>
    </div>
    
    <div class="endpoint">
        <h2>Verificar estado del servicio</h2>
        <p><strong>Endpoint:</strong> <code>GET /health</code></p>
        <p><strong>Descripción:</strong> Verifica si el servicio está funcionando correctamente.</p>
    </div>
    
    <div class="endpoint">
        <h2>Configuración actual</h2>
        <p><strong>Endpoint:</strong> <code>GET /config</code></p>
        <p><strong>Descripción:</strong> Muestra la configuración actual de los modelos en uso.</p>
    </div>
    
    <div class="endpoint">
        <h2>Proveedores disponibles</h2>
        <p><strong>Endpoint:</strong> <code>GET /providers</code></p>
        <p><strong>Descripción:</strong> Lista todos los proveedores de modelos disponibles.</p>
    </div>
    
    <h2>Configuración Actual:</h2>
    <table>
        <tr>
            <th>Componente</th>
            <th>Proveedor</th>
            <th>Modelo</th>
        </tr>
        <tr>
            <td>LLM</td>
            <td>{llm_provider}</td>
            <td>{llm_model}</td>
        </tr>
        <tr>
            <td>Embeddings</td>
            <td>{embedding_provider}</td>
            <td>{embedding_model}</td>
        </tr>
    </table>
    
    <script>
        function testApi() {{
            const query = document.getElementById('query').value.trim();
            if (!query) return;
            
            const resultDiv = document.getElementById('result');
            const loadingSpan = document.getElementById('loading');
            
            resultDiv.innerHTML = '';
            loadingSpan.style.display = 'inline';
            
            fetch('/ask', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify({{ message: query }})
            }})
            .then(response => response.json())
            .then(data => {{
                loadingSpan.style.display = 'none';
                
                // Format the result
                let result = '<h3>Respuesta:</h3><div style="background: #f0f7ff; padding: 15px; border-radius: 5px; margin-bottom: 15px;">';
                result += data.answer.replace(/\\n/g, '<br>');
                result += '</div>';
                
                if (data.sources && data.sources.length > 0) {{
                    result += '<h3>Fuentes:</h3><ul>';
                    data.sources.forEach(source => {{
                        result += '<li>';
                        if (source.metadata && source.metadata.title) {{
                            result += '<strong>' + source.metadata.title + '</strong><br>';
                        }}
                        if (source.metadata && source.metadata.url) {{
                            result += '<a href="' + source.metadata.url + '" target="_blank">' + source.metadata.url + '</a><br>';
                        }}
                        if (source.content) {{
                            result += '<small>' + source.content.substring(0, 250) + '...</small>';
                        }}
                        result += '</li>';
                    }});
                    result += '</ul>';
                }}
                
                resultDiv.innerHTML = result;
            }})
            .catch(error => {{
                loadingSpan.style.display = 'none';
                resultDiv.innerHTML = '<div style="color: red; background: #fff0f0; padding: 15px; border-radius: 5px;">Error: ' + error.message + '</div>';
            }});
        }}
        
        // Allow pressing Enter to submit
        document.getElementById('query').addEventListener('keydown', function(event) {{
            if (event.key === 'Enter') {{
                event.preventDefault();
                testApi();
            }}
        }});
    </script>
</body>
</html>
"""

def get_rag_chain():
    global rag_pipeline
    if rag_pipeline is None:
        # Initialize with configured providers and models
        rag_pipeline = load_rag_chain(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
    return rag_pipeline

@app.route('/')
def home():
    """Página principal con interfaz web para testing del API"""
    return render_template_string(API_DOCS.format(
        llm_provider=llm_provider,
        llm_model=llm_model or "default",
        embedding_provider=embedding_provider,
        embedding_model=embedding_model or "default"
    ))

@app.route('/api-info')
def api_info():
    """Documentación de la API en formato JSON"""
    docs = {
        "title": "Asistente RAG - Trámites de Córdoba",
        "description": "API de Generación Aumentada por Recuperación para consultas gubernamentales",
        "version": "2.0.0",
        "improvements": [
            "✅ Preprocessing optimizado para Transformers (sin remoción de stopwords/tildes)",
            "✅ Chunking inteligente basado en modelo de embedding",
            "✅ Validación automática de consistencia de modelos",
            "✅ Metadatos detallados para troubleshooting",
            "✅ Documentación clara de especificación de modelos"
        ],
        "endpoints": {
            "GET /": "Interfaz web para testing",
            "GET /api-info": "Esta documentación en JSON",
            "GET /health": "Estado del servicio y validación de modelos",
            "GET /config": "Configuración completa del sistema (LLM + Embeddings + Chunking)",
            "GET /providers": "Proveedores disponibles de modelos",
            "GET /system-info": "Información detallada del sistema para debugging",
            "POST /ask": "Realizar consulta (incluye info del sistema en respuesta)"
        },
        "model_architecture": {
            "embedding_model": {
                "purpose": "Búsqueda de similaridad en vector store",
                "current": f"{embedding_provider}:{embedding_model or 'default'}",
                "note": "DEBE ser el mismo para indexación y consulta"
            },
            "llm_model": {
                "purpose": "Generación de respuestas basadas en contexto recuperado", 
                "current": f"{llm_provider}:{llm_model or 'default'}",
                "note": "Independiente del modelo de embedding"
            }
        },
        "usage": {
            "curl_example": f"""
curl -X POST http://localhost:5000/ask \\
     -H 'Content-Type: application/json' \\
     -d '{{"message": "¿Qué necesito para obtener un certificado de antecedentes?"}}'
            """.strip(),
            "response_includes": [
                "answer: Respuesta generada por el LLM",
                "sources: Documentos fuente recuperados",
                "system_info: Información del sistema (modelos, chunking, etc.)"
            ]
        }
    }
    return jsonify(docs)

@app.route('/health')
def health():
    """Estado del servicio con validación completa de modelos"""
    global rag_pipeline
    
    try:
        # Intentar inicializar o verificar el pipeline
        if rag_pipeline is None:
            rag_pipeline = load_rag_chain(
                llm_provider=llm_provider,
                llm_model=llm_model,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            )
        
        # Obtener información del sistema
        system_info = rag_pipeline.get_system_info()
        
        return jsonify({
            "status": "healthy",
            "message": "✅ Servicio operativo con validación de modelos exitosa",
            "model_validation": "passed",
            "embedding_consistency": "verified",
            "system_ready": True,
            "models": {
                "embedding": {
                    "provider": system_info["embedding_model_info"]["provider"],
                    "model": system_info["embedding_model_info"]["model"],
                    "usage": system_info["embedding_model_info"]["usage"]
                },
                "llm": {
                    "provider": system_info["llm_model_info"]["provider"], 
                    "model": system_info["llm_model_info"]["model"],
                    "usage": system_info["llm_model_info"]["usage"]
                }
            },
            "performance_metrics": {
                "total_chunks": system_info["total_chunks"],
                "chunk_strategy": system_info["chunking_strategy"],
                "preprocessing": system_info["preprocessing_strategy"]
            }
        })
        
    except ModelValidationError as e:
        return jsonify({
            "status": "unhealthy",
            "message": "❌ Error de validación de modelos",
            "error": str(e),
            "model_validation": "failed",
            "system_ready": False,
            "action_required": "Verificar configuración de modelos o recrear índice"
        }), 500
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "message": "❌ Error en inicialización del servicio",
            "error": str(e),
            "system_ready": False
        }), 500

@app.route('/config')
def config():
    """Configuración completa del sistema con especificación detallada de modelos"""
    global rag_pipeline
    
    base_config = {
        "runtime_configuration": {
            "llm_provider": llm_provider,
            "llm_model": llm_model or f"{llm_provider}_default",
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model or f"{embedding_provider}_default"
        },
        "model_specifications": {
            "embedding_model": {
                "purpose": "Vector similarity search in document index",
                "responsibility": "Convert text queries and documents to vectors for retrieval",
                "consistency_requirement": "MUST match model used during indexing",
                "provider": embedding_provider,
                "model": embedding_model or "default"
            },
            "llm_model": {
                "purpose": "Generate responses based on retrieved context",
                "responsibility": "Process retrieved documents + user query → final answer",
                "independence": "Can be different from embedding model",
                "provider": llm_provider,
                "model": llm_model or "default"
            }
        }
    }
    
    # Agregar información del índice si el pipeline está disponible
    if rag_pipeline:
        try:
            system_info = rag_pipeline.get_system_info()
            base_config["index_configuration"] = {
                "chunking_strategy": system_info["chunking_strategy"],
                "preprocessing_strategy": system_info["preprocessing_strategy"],
                "total_chunks": system_info["total_chunks"],
                "embedding_model_verified": system_info["embedding_model_info"]
            }
        except Exception as e:
            base_config["index_configuration"] = {
                "status": "error",
                "message": str(e)
            }
    
    return jsonify(base_config)

@app.route('/system-info')
def system_info():
    """Información detallada del sistema para debugging y análisis"""
    global rag_pipeline
    
    if not rag_pipeline:
        try:
            rag_pipeline = load_rag_chain(
                llm_provider=llm_provider,
                llm_model=llm_model,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            )
        except Exception as e:
            return jsonify({
                "error": "Pipeline no disponible",
                "message": str(e),
                "action": "Verificar configuración y recrear índice si es necesario"
            }), 500
    
    try:
        detailed_info = rag_pipeline.get_system_info()
        
        # Agregar información adicional de debugging
        detailed_info["pipeline_status"] = {
            "initialized": True,
            "vector_store_loaded": True,
            "models_validated": True
        }
        
        detailed_info["technical_details"] = {
            "chunking_rationale": "Tamaño optimizado basado en ventana de contexto del modelo de embedding",
            "preprocessing_rationale": "Sin remoción de stopwords/tildes - apropiado para Transformers",
            "validation_strategy": "Metadatos guardados durante indexación para verificar consistencia"
        }
        
        return jsonify(detailed_info)
        
    except Exception as e:
        return jsonify({
            "error": "Error obteniendo información del sistema",
            "message": str(e)
        }), 500

@app.route('/providers')
def providers():
    """Información actualizada de proveedores disponibles"""
    return jsonify({
        "llm_providers": {
            "openai": {
                "name": "OpenAI", 
                "description": "GPT models (requires API key)",
                "models": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
                "usage": "Generación de respuestas finales"
            },
            "ollama": {
                "name": "Ollama",
                "description": "Local models through Ollama",
                "models": ["mistral", "llama2", "codellama"],
                "usage": "Generación local sin dependencias externas"
            },
            "huggingface": {
                "name": "HuggingFace",
                "description": "HuggingFace models",
                "models": ["google/flan-t5-xxl", "microsoft/DialoGPT-medium"],
                "usage": "Modelos open-source"
            }
        },
        "embedding_providers": {
            "openai": {
                "name": "OpenAI",
                "description": "OpenAI embedding models",
                "models": ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
                "optimal_chunk_size": "512 tokens",
                "usage": "Búsqueda de similaridad en vector store"
            },
            "ollama": {
                "name": "Ollama", 
                "description": "Local embedding models",
                "models": ["nomic-embed-text", "all-minilm"],
                "optimal_chunk_size": "256-384 tokens",
                "usage": "Embeddings locales sin API"
            },
            "huggingface": {
                "name": "HuggingFace",
                "description": "HuggingFace embedding models",
                "models": ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
                "optimal_chunk_size": "256-512 tokens",
                "usage": "Embeddings open-source"
            }
        },
        "configuration_note": "El modelo de embedding DEBE ser consistente entre indexación y consulta. El LLM puede ser independiente.",
        "chunking_optimization": "Los tamaños de chunk se ajustan automáticamente según el modelo de embedding elegido."
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Realiza una consulta al sistema RAG"""
    global rag_pipeline
    
    # Asegurarse de que el pipeline esté inicializado
    if rag_pipeline is None:
        try:
            rag_pipeline = load_rag_chain(
                llm_provider=llm_provider,
                llm_model=llm_model,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            )
        except ModelValidationError as e:
            return jsonify({"error": "Error de validación de modelo", "message": str(e)}), 400
        except Exception as e:
            return jsonify({"error": "Error de inicialización del pipeline", "message": str(e)}), 500

    if not request.json or 'message' not in request.json:
        return jsonify({"error": "La consulta debe estar en formato JSON con la clave 'message'"}), 400

    question = request.json['message']
    
    try:
        result = rag_pipeline(question)
        
        # Formatear respuesta para el cliente
        return jsonify({
            "answer": result.get("result", "No se pudo generar una respuesta."),
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result.get("source_documents", [])
            ],
            "system_info": result.get("system_info", {})
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Error al procesar la consulta",
            "message": str(e)
        }), 500

def parse_args():
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="""
    Servidor Flask para el sistema RAG de Trámites de Córdoba.
    
    La configuración de modelos se toma en el siguiente orden de prioridad:
    1. Argumentos de línea de comandos (ej: --llm-provider openai)
    2. Variables de entorno (ej: export LLM_PROVIDER=openai)
    3. Archivo `rag_flask/config/rag_config.json`
    """)
    
    # Cargar defaults desde el entorno o dejarlos en None para que rag_chain use el JSON
    parser.add_argument("--llm-provider", type=str, default=os.getenv("LLM_PROVIDER"),
                        help="Proveedor del LLM para generación (e.g., 'openai', 'ollama').")
    parser.add_argument("--llm-model", type=str, default=os.getenv("LLM_MODEL"),
                        help="Modelo específico del LLM.")
    parser.add_argument("--embedding-provider", type=str, default=os.getenv("EMBEDDING_PROVIDER"),
                        help="Proveedor del modelo de embedding para búsqueda.")
    parser.add_argument("--embedding-model", type=str, default=os.getenv("EMBEDDING_MODEL"),
                        help="Modelo específico de embedding.")
    parser.add_argument("--port", type=int, default=5000, help="Puerto para ejecutar el servidor.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Asignar argumentos a las variables globales
    llm_provider = args.llm_provider
    llm_model = args.llm_model
    embedding_provider = args.embedding_provider
    embedding_model = args.embedding_model

    print("🚀 Iniciando servidor RAG...")
    print(f"   - LLM: {llm_provider or 'default'}:{llm_model or 'default'}")
    print(f"   - Embeddings: {embedding_provider or 'default'}:{embedding_model or 'default'}")
    print(f"   - Puerto: {args.port}")

    app.run(host='0.0.0.0', port=args.port, debug=True)