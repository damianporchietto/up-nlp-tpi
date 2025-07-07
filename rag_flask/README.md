# Asistente RAG de Trámites de Córdoba - Versión 2.0

Servicio de **Generación Aumentada por Recuperación (RAG)** para consultas sobre trámites y servicios gubernamentales de la Provincia de Córdoba. 

## 🚀 **Mejoras Implementadas (v2.0)**

### ✅ **Correcciones Críticas Basadas en Observaciones Académicas**

1. **🔧 Preprocesamiento Optimizado para Transformers**
   - ❌ **ELIMINADO**: Remoción de stopwords (contraproducente para modelos modernos)
   - ❌ **ELIMINADO**: Remoción de tildes y acentos (importante para español)
   - ✅ **MANTENIDO**: Solo normalización básica apropiada para Transformers

2. **📏 Chunking Inteligente y Optimizado**
   - ✅ **Tamaños adaptativos** basados en modelo de embedding (256-512 tokens)
   - ✅ **Overlap optimizado** (10% del chunk size)
   - ✅ **Justificación técnica** de parámetros basada en investigación RAG 2024
   - ✅ **Consideración de ventana de contexto** de modelos

3. **🔒 Validación de Consistencia de Modelos**
   - ✅ **Metadatos automáticos** guardados durante indexación
   - ✅ **Validación obligatoria** entre modelo de indexación y consulta
   - ✅ **Detección de inconsistencias** con mensajes de error claros

4. **📊 Especificación Clara de Modelos**
   - ✅ **Documentación explícita** de qué modelo se usa para qué
   - ✅ **Endpoints informativos** con detalles del sistema
   - ✅ **Arquitectura claramente definida**: Embedding vs LLM

5. **🎯 Framework de Evaluación Robusto**
   - ✅ **Métricas cuantitativas** (Recall@K, Precision@K, MRR, similitud semántica)
   - ✅ **Dataset de evaluación expandible**
   - ✅ **Benchmarking automático** de configuraciones
   - ✅ **Reportes con análisis** y recomendaciones

## 📁 **Estructura del Proyecto**

```
rag_flask/
├── app.py                    # 🌐 API Flask con validación de modelos
├── rag_chain.py              # 🧠 Pipeline RAG con validación automática
├── ingest.py                 # 📥 Ingesta con chunking optimizado
├── preprocessing.py          # 🔧 Preprocesamiento para Transformers
├── model_providers.py        # 🤖 Gestión de proveedores de modelos
├── evaluation.py             # 📊 Framework de evaluación comprehensivo
├── requirements.txt          # 📦 Dependencias
├── docs/                     # 📄 Documentos fuente (JSON por ministerio)
├── storage/                  # 💾 Índice FAISS + metadatos
│   └── index_metadata.json   # 🔍 Validación de consistencia
└── test/                     # 🧪 Tests y evaluaciones
```

## 🏗️ **Arquitectura de Modelos - Especificación Técnica**

### **Modelo de Embedding** (Búsqueda de Similaridad)
- **Propósito**: Convertir texto a vectores para búsqueda en índice
- **Uso**: Indexación de documentos + consultas de usuario
- **Requisito CRÍTICO**: Debe ser el mismo modelo para ambos procesos
- **Configuración**: `--embedding-provider` y `--embedding-model`

### **Modelo LLM** (Generación de Respuestas)
- **Propósito**: Generar respuesta final basada en contexto recuperado
- **Uso**: Solo para generación de texto
- **Independencia**: Puede ser diferente del modelo de embedding
- **Configuración**: `--llm-provider` y `--llm-model`

## 🚀 **Inicio Rápido**

### 1. Configuración del Entorno

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
echo 'OPENAI_API_KEY=sk-...' >> .env
```

### 2. Crear Índice Vectorial (Indexación)

```bash
# Con OpenAI (recomendado para producción)
python ingest.py --provider openai --model text-embedding-3-large

# Con Ollama (para desarrollo local)
python ingest.py --provider ollama --model nomic-embed-text

# El chunking se optimiza automáticamente según el modelo
```

### 3. Ejecutar API (Consultas)

```bash
# IMPORTANTE: Usar los mismos modelos que en indexación
python app.py \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large \
  --llm-provider openai \
  --llm-model gpt-4o-mini
```

## 🔍 **Validación Automática de Modelos**

El sistema ahora **valida automáticamente** que uses el mismo modelo de embedding:

```bash
# ✅ Correcto: Modelos consistentes
python ingest.py --provider openai --model text-embedding-3-large
python app.py --embedding-provider openai --embedding-model text-embedding-3-large

# ❌ Error detectado automáticamente
python ingest.py --provider openai --model text-embedding-3-large  
python app.py --embedding-provider ollama --embedding-model nomic-embed-text
# → ModelValidationError: Inconsistencia detectada
```

## 📊 **Evaluación y Benchmarking**

### Evaluación Simple

```bash
# Evaluar configuración actual
python evaluation.py --single-eval

# Resultados incluyen:
# - Score general (0-1)
# - Recall@3, Precision@3, MRR
# - Similitud semántica promedio
# - Tiempo de respuesta promedio
```

### Benchmark de Configuraciones

```bash
# Crear archivo de configuraciones a comparar
cat > benchmark_configs.json << EOF
[
  {
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
    "embedding_provider": "openai", 
    "embedding_model": "text-embedding-3-large"
  },
  {
    "llm_provider": "ollama",
    "llm_model": "mistral",
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text"
  }
]
EOF

# Ejecutar benchmark
python evaluation.py --config-file benchmark_configs.json
```

## 🌐 **Endpoints de la API**

### Información del Sistema
```bash
# Documentación completa con arquitectura
curl http://localhost:5000/

# Estado con validación de modelos
curl http://localhost:5000/health

# Configuración detallada del sistema
curl http://localhost:5000/config

# Información técnica para debugging
curl http://localhost:5000/system-info
```

### Consultas
```bash
curl -X POST http://localhost:5000/ask \
     -H 'Content-Type: application/json' \
     -d '{"message": "¿Qué necesito para obtener un certificado de antecedentes?"}'
```

**Respuesta incluye**:
- `answer`: Respuesta generada
- `sources`: Documentos fuente con snippets
- `system_info`: Modelos usados y métricas del sistema

## ⚙️ **Configuración de Chunking Optimizada**

El sistema ahora **ajusta automáticamente** el chunking según el modelo:

| Modelo de Embedding | Chunk Size Óptimo | Justificación |
|--------------------|--------------------|---------------|
| `text-embedding-3-large` | 512 tokens (~2300 chars) | Ventana de contexto optimizada |
| `nomic-embed-text` | 384 tokens (~1700 chars) | Modelo local balanceado |
| `all-MiniLM-L6-v2` | 256 tokens (~1150 chars) | Modelo liviano eficiente |

**Overlap**: 10% del chunk size (balance entre contexto y eficiencia)

## 🧪 **Estrategia de Preprocesamiento Actualizada**

### ❌ **LO QUE NO HACEMOS (Contraproducente para Transformers)**
- Remover stopwords ("el", "la", "de", etc.)
- Quitar tildes y acentos 
- Lemmatización agresiva

### ✅ **LO QUE SÍ HACEMOS (Apropiado para Transformers)**
- Normalización de espacios múltiples
- Conversión a minúsculas para consistencia
- Eliminación solo de caracteres verdaderamente problemáticos
- Preservación del contexto sintáctico completo

## 🔧 **Proveedores de Modelos Soportados**

### OpenAI (Recomendado para Producción)
```bash
# Embeddings
--embedding-provider openai --embedding-model text-embedding-3-large

# LLM
--llm-provider openai --llm-model gpt-4o-mini
```

### Ollama (Desarrollo Local)
```bash
# Instalar Ollama primero: https://ollama.ai
ollama pull nomic-embed-text
ollama pull mistral

# Usar en el sistema
--embedding-provider ollama --embedding-model nomic-embed-text
--llm-provider ollama --llm-model mistral
```

### HuggingFace (Open Source)
```bash
# Requiere instalación adicional
pip install transformers torch sentence-transformers

# Configuración
--embedding-provider huggingface --embedding-model BAAI/bge-large-en-v1.5
--llm-provider huggingface --llm-model google/flan-t5-xxl
```

## 📈 **Monitoreo y Debugging**

### Logs del Sistema
El sistema ahora proporciona logs detallados:
```
🔧 Inicializando RAG Pipeline:
   • LLM (generación): openai:gpt-4o-mini
   • Embeddings (búsqueda): openai:text-embedding-3-large
📏 Configuración de chunking optimizada para openai:text-embedding-3-large
   • Tamaño objetivo: 512 tokens (~2304 caracteres)
   • Overlap: 51 tokens (~230 caracteres, 10.0%)
✅ Validación de modelo exitosa
🚀 Pipeline RAG inicializado correctamente
```

### Métricas de Performance
- **Tiempo de respuesta promedio**: Incluido en `/system-info`
- **Chunks recuperados por consulta**: Monitoreado automáticamente
- **Diversidad de fuentes**: Métrica de calidad del retrieval

## 🚨 **Solución de Problemas Comunes**

### Error: Inconsistencia de Modelos
```
❌ Inconsistencia de modelo de embedding:
   • Índice creado con: openai:text-embedding-3-large  
   • Consulta usando: ollama:nomic-embed-text
```
**Solución**: Usar el mismo modelo o recrear índice
```bash
python ingest.py --provider ollama --model nomic-embed-text
```

### Performance Lenta
1. **Verificar modelo**: Modelos locales pueden ser más lentos
2. **Revisar chunking**: Chunks muy grandes afectan performance
3. **Optimizar retrieval**: Reducir `k` en configuración del retriever

### Calidad de Respuestas Baja
1. **Evaluar sistema**: `python evaluation.py --single-eval`
2. **Revisar chunking**: Evaluar si preserva contexto semántico
3. **Probar modelos**: Benchmark diferentes configuraciones

## 📚 **Dataset de Evaluación**

El sistema incluye un dataset de evaluación expandible (`evaluation_dataset.json`):

```json
[
  {
    "query": "¿Qué necesito para obtener un certificado de antecedentes?",
    "expected_answer": "Necesitas DNI en perfecto estado...",
    "relevant_doc_ids": ["certificado_antecedentes"],
    "category": "certificados",
    "difficulty": "easy"
  }
]
```

**Para ampliar**: Agrega más consultas con respuestas esperadas y documentos relevantes.

## 🔬 **Investigación y Fundamentación Técnica**

### Chunking Basado en Investigación
- **Referencia**: "Evaluating the ideal chunk size for a rag system" (Vectorize, 2024)
- **Hallazgo clave**: 256-512 tokens óptimo para embeddings modernos
- **Implementación**: Configuración adaptativa por modelo

### Preprocesamiento para Transformers
- **Principio**: Modelos entrenados con texto natural completo
- **Evidence**: Stopwords aportan contexto sintáctico valioso
- **Resultado**: Mayor precisión semántica en embeddings

## 🤝 **Contribución y Desarrollo**

### Agregar Nuevos Documentos
```bash
# 1. Agregar archivos JSON a docs/MINISTERIO/
# 2. Recrear índice
python ingest.py

# El sistema detecta automáticamente nuevos archivos
```

### Desarrollo de Features
1. **Fork el repositorio**
2. **Crear branch**: `git checkout -b feature/nueva-funcionalidad`
3. **Ejecutar tests**: `python evaluation.py --single-eval`
4. **Commit y push**: Incluir resultados de evaluación
5. **Pull Request**: Con métricas de performance

---

## 📋 **Resumen de Correcciones Implementadas**

| Observación Académica | ✅ Corrección Implementada |
|----------------------|---------------------------|
| Preprocesamiento contraproducente | Eliminado stopwords/tildes, optimizado para Transformers |
| Chunking sin justificación | Configuración adaptativa basada en investigación 2024 |
| Falta validación de modelos | Metadatos automáticos + validación obligatoria |
| Especificación poco clara | Documentación exhaustiva de arquitectura |
| Evaluación no robusta | Framework con métricas cuantitativas + benchmarking |
| Inconsistencias documentación | Alineación completa código-docs-funcionalidad |

**Resultado**: Sistema RAG robusto, bien documentado y técnicamente fundamentado, listo para evaluación académica y uso en producción.