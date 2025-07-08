# ⚙️ Guía de Configuración de Modelos (`rag_config.json`)

Este documento explica cómo configurar los modelos de Lenguaje (LLM) y de Embedding en el archivo `rag_config.json`. La flexibilidad de este sistema te permite combinar diferentes proveedores según tus necesidades de rendimiento, costo y privacidad.

---

## 🧠 1. Configuración del LLM (`llm_config`)

El LLM es el "cerebro" que **genera las respuestas**. Se utiliza únicamente en la fase final del proceso para sintetizar la información recuperada.

-   `"default_provider"`: Define qué servicio se usará para el LLM.
-   `"default_model"`: Especifica el modelo exacto a utilizar.
-   `"temperature"`: Controla la "creatividad" de la respuesta. Para un asistente basado en hechos, `0` es ideal.

### Opciones de Proveedores:

#### a) `openai`
-   **Descripción**: Modelos de alto rendimiento con costo por uso. Requiere una `OPENAI_API_KEY` en el archivo `.env`.
-   **Ejemplos de Modelos (`default_model`)**:
    -   `"gpt-4o-mini"`: **(Recomendado)** Excelente balance entre costo, velocidad y calidad.
    -   `"gpt-4-turbo"`: Modelo de máxima calidad, más costoso.
    -   `"gpt-3.5-turbo"`: Opción más económica y rápida.

#### b) `ollama`
-   **Descripción**: Permite ejecutar modelos de código abierto **localmente** en tu propia máquina. Ideal para privacidad y experimentación sin costo. Debes tener [Ollama](https://ollama.com/) instalado y los modelos descargados (`ollama run llama3`).
-   **Ejemplos de Modelos (`default_model`)**:
    -   `"llama3"`: Modelo de última generación de Meta. Muy capaz.
    -   `"mistral"`: Excelente modelo de Mistral AI.
    -   `"gemma"`: Familia de modelos abiertos de Google.

#### c) `huggingface`
-   **Descripción**: Permite cargar modelos desde el Hugging Face Hub. Más complejo de configurar, generalmente requiere dependencias adicionales y más recursos computacionales.
-   **Ejemplos de Modelos (`default_model`)**:
    -   `"google/flan-t5-large"`
    -   `"HuggingFaceH4/zephyr-7b-beta"`

---

## 🔍 2. Configuración de Embeddings (`embedding_config`)

El modelo de embedding es responsable de **entender el significado del texto** para la búsqueda. Convierte tanto los documentos como las preguntas del usuario en vectores numéricos.

> **⚠️ ¡ADVERTENCIA CRÍTICA!**
> El modelo de embedding que uses para la **ingesta (`ingest.py`)** y para las **consultas (`app.py`)** debe ser **EXACTAMENTE EL MISMO**. Nuestro sistema tiene una validación automática para prevenir este error, pero es fundamental que tu configuración sea consistente. Si cambias el modelo de embedding, **debes borrar la carpeta `/storage` y volver a ejecutar la ingesta.**

### Opciones de Proveedores:

#### a) `openai`
-   **Descripción**: Modelos de embedding de muy alta calidad y optimizados.
-   **Ejemplos de Modelos (`default_model`)**:
    -   `"text-embedding-3-large"`: El más potente.
    -   `"text-embedding-3-small"`: **(Recomendado)** Gran balance rendimiento/costo.
    -   `"text-embedding-ada-002"`: Modelo de la generación anterior, más económico.

#### b) `ollama`
-   **Descripción**: Modelos de embedding que se ejecutan localmente.
-   **Ejemplos de Modelos (`default_model`)**:
    -   `"nomic-embed-text"`: **(Recomendado para local)** Modelo de alto rendimiento diseñado específicamente para embeddings.
    -   `"all-minilm"`: Un modelo más ligero y rápido.

#### c) `huggingface`
-   **Descripción**: Acceso a miles de modelos de embedding del Hub.
-   **Ejemplos de Modelos (`default_model`)**:
    -   `"BAAI/bge-large-en-v1.5"`
    -   `"sentence-transformers/all-MiniLM-L6-v2"`

---

## 🛠️ 3. Ejemplos de Archivos `rag_config.json`

#### Escenario 1: Máximo Rendimiento (usando OpenAI)
```json
{
  "llm_config": {
    "default_provider": "openai",
    "default_model": "gpt-4o-mini"
  },
  "embedding_config": {
    "default_provider": "openai",
    "default_model": "text-embedding-3-small"
  },
  // ...
}
```

#### Escenario 2: Totalmente Local y Privado (usando Ollama)
```json
{
  "llm_config": {
    "default_provider": "ollama",
    "default_model": "llama3"
  },
  "embedding_config": {
    "default_provider": "ollama",
    "default_model": "nomic-embed-text"
  },
  // ...
}
```

#### Escenario 3: Híbrido (LLM potente, embeddings locales)
```json
{
  "llm_config": {
    "default_provider": "openai",
    "default_model": "gpt-4o-mini"
  },
  "embedding_config": {
    "default_provider": "ollama",
    "default_model": "nomic-embed-text"
  },
  // ...
}
```
*Recuerda que si usas esta configuración, debes ejecutar `python ingest.py --provider ollama --model nomic-embed-text` y luego `python app.py --llm-provider openai`.* 