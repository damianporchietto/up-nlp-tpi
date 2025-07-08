# RAG Solution: Asistente Inteligente para Trámites Gubernamentales

## 📜 Resumen Ejecutivo

Este documento ofrece una explicación completa y detallada de la solución de **Generación Aumentada por Recuperación (RAG)** implementada para crear un asistente de IA especializado en los trámites gubernamentales de la Provincia de Córdoba, Argentina.

El sistema está diseñado para ser **robusto, configurable y académicamente riguroso**, abordando los desafíos comunes de los sistemas RAG y aplicando las mejores prácticas de la industria.

**Capacidades Clave:**
- **Respuestas Basadas en Evidencia**: Contesta preguntas utilizando únicamente la información oficial cargada en su base de conocimiento.
- **Configuración Flexible**: Permite cambiar fácilmente los modelos de lenguaje (LLM) y de embedding (OpenAI, Ollama, HuggingFace).
- **Validación Automática**: Previene errores críticos al garantizar que los modelos utilizados para indexar y consultar sean consistentes.
- **Optimización Inteligente**: Ajusta automáticamente sus parámetros internos (como el *chunking*) según el modelo seleccionado para maximizar la calidad.

---

## 🧐 1. ¿Qué es un Sistema RAG?

Un sistema de **Generación Aumentada por Recuperación (RAG)** es una arquitectura de inteligencia artificial que combina lo mejor de dos mundos:

1.  **Modelos de Lenguaje Pre-entrenados (LLMs)**: Como GPT-4 o Llama3, que son excelentes para entender y generar texto, pero su conocimiento está "congelado" en el momento en que fueron entrenados y pueden "alucinar" o inventar información.
2.  **Bases de Conocimiento Externas**: Colecciones de documentos privados y actualizados (PDFs, JSONs, etc.) que contienen información específica y verídica.

**El Proceso RAG en 3 Pasos:**

1.  **🔍 Recuperar (Retrieve)**: Cuando un usuario hace una pregunta, el sistema no se la pasa directamente al LLM. Primero, busca en su base de conocimiento (en nuestro caso, un **Vector Store**) los fragmentos de texto más relevantes para la pregunta.
2.  **➕ Aumentar (Augment)**: El sistema toma los fragmentos recuperados (el "contexto") y los inserta en un *prompt* junto con la pregunta original del usuario.
3.  **✍️ Generar (Generate)**: Finalmente, le entrega este *prompt aumentado* al LLM, dándole la instrucción: "Responde a esta pregunta basándote *únicamente* en este contexto".

Esto obliga al LLM a basar su respuesta en los documentos oficiales, reduciendo drásticamente las alucinaciones y permitiéndole responder sobre datos que no estaban en su entrenamiento original.

---

## ⚙️ 2. Arquitectura de Nuestra Solución

Nuestro sistema sigue la arquitectura RAG clásica, pero con optimizaciones y validaciones clave en cada etapa.

```mermaid
graph TD
    subgraph Fase 1: Ingesta de Datos (Offline)
        A["📄 Documentos JSON<br/>(en /docs)"] --> B["🔧 Preprocessing<br/>(Optimizado para Transformers)"];
        B --> C["📦 Chunking Adaptativo<br/>(Fragmentación Inteligente)"];
        C --> D["🧠 Conversión a Vectores<br/>(Modelo de Embedding)"];
        D --> E["💾 Vector Store (FAISS)<br/>(Guardado en /storage)"];
        E --> F["📝 Metadatos<br/>(index_metadata.json)"];
    end

    subgraph Fase 2: Ciclo de Consulta (Online)
        G["❓ Pregunta del Usuario"] --> H["🧠 Conversión a Vector<br/>(Mismo Modelo de Embedding)"];
        H --> I{🔍 Búsqueda de Similitud};
        E --> I;
        I --> J["📚 Contexto Recuperado<br/>(Top-K Chunks Relevantes)"];
        G --> K{➕ Prompt Aumentado};
        J --> K;
        K --> L["🤖 LLM (Generación)<br/>(gpt-4o-mini, llama3, etc.)"];
        L --> M["✅ Respuesta Final + Fuentes"];
    end
```

### Componentes Clave:

#### **a. Ingesta y Preprocesamiento (`ingest.py`, `preprocessing.py`)**
El proceso comienza con los documentos JSON ubicados en la carpeta `rag_flask/docs/`.

-   **Preprocessing Transformer-Optimizado**: A diferencia de técnicas de NLP más antiguas, **no eliminamos stopwords (palabras comunes como "el", "de") ni acentos**. Los modelos Transformer modernos utilizan estas características para entender el contexto y la semántica. Nuestra limpieza se centra en normalizar espacios y eliminar caracteres problemáticos sin destruir la estructura del lenguaje.
-   **Script de Ingesta**: El script `ingest.py` orquesta todo el proceso de preparación de datos, desde leer los archivos hasta construir la base de datos vectorial final.

#### **b. Chunking Adaptativo (Fragmentación Inteligente)**
Los documentos largos no pueden ser procesados directamente por los LLMs. Por eso, los dividimos en fragmentos más pequeños o "chunks".

-   **Estrategia**: Usamos `RecursiveCharacterTextSplitter`, que intenta dividir el texto por párrafos, saltos de línea y frases, preservando así la cohesión semántica.
-   **Optimización Automática**: El tamaño de estos chunks no es fijo. El sistema lo ajusta automáticamente (`get_optimal_chunk_config`) basándose en el modelo de embedding que se esté utilizando. Esto es crucial, ya que cada modelo tiene una "ventana de contexto" óptima.
-   **Overlap (Solapamiento)**: Cada chunk comparte un pequeño porcentaje de contenido (10-15%) con el chunk anterior y siguiente. Esto evita que una idea importante se corte justo en la frontera entre dos fragmentos.

#### **c. Embeddings y Vector Store (`model_providers.py`, `storage/`)**
-   **Embeddings**: Son representaciones numéricas (vectores) del texto. Un modelo de embedding convierte cada chunk en un vector de forma que los chunks con significados similares tengan vectores cercanos en el espacio.
-   **Vector Store**: Utilizamos **FAISS** (de Facebook AI), una biblioteca ultra-eficiente para buscar vectores similares. Nuestra base de datos vectorial se guarda en la carpeta `rag_flask/storage/`.

#### **d. Cadena de Consulta (`rag_chain.py`)**
Esta es la lógica central que se ejecuta cuando un usuario realiza una consulta.

-   **Recuperación**: Se vectoriza la pregunta del usuario y se usa FAISS para encontrar los `k` chunks más similares (por defecto, `k=4`).
-   **Aumentación (Prompting)**: Se utiliza una plantilla de prompt (`PROMPT_TEMPLATE`) que instruye claramente al LLM sobre su rol y le obliga a usar el contexto proporcionado.
-   **Generación**: El LLM sintetiza la información de los chunks recuperados en una respuesta coherente y bien redactada.

---

## 💡 3. Decisiones Técnicas Clave

Este sistema no es un RAG genérico. Se han implementado varias mejoras cruciales para responder a problemas del mundo real.

#### **1. Configuración Centralizada (`config/rag_config.json`)**
Para evitar tener parámetros "hardcodeados" en el código, toda la configuración principal reside en `rag_flask/config/rag_config.json`. Esto permite modificar el comportamiento del sistema (cambiar modelos, prompts, etc.) sin tocar una sola línea de código Python.

#### **2. Validación de Consistencia de Modelos (¡Crítico!)**
Un error catastrófico en los sistemas RAG ocurre cuando se indexan los documentos con un modelo de embedding (ej. `text-embedding-3-large` de OpenAI) y luego se intenta hacer una consulta con otro (ej. `nomic-embed-text` de Ollama). Los vectores son incompatibles y los resultados de búsqueda son basura.

-   **Nuestra Solución**: Durante la ingesta (`ingest.py`), guardamos un archivo de metadatos (`index_metadata.json`) que registra qué modelo se usó. Al iniciar la consulta (`rag_chain.py`), el sistema **valida** que el modelo actual coincida con el de los metadatos. Si no es así, lanza un error claro y explícito, evitando fallos silenciosos.

---

## 🚀 4. Guía de Uso Rápido

Sigue estos pasos para poner en marcha el sistema.

#### **Paso 0: Configuración**
Antes de nada, revisa y ajusta `rag_flask/config/rag_config.json`. Aquí puedes definir los modelos por defecto.

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
  // ... resto de la configuración
}
```

Si usas OpenAI, asegúrate de tener tu clave de API en un archivo `.env`:
```
OPENAI_API_KEY="sk-..."
```

#### **Paso 1: Ingesta de Datos**
Este comando procesará los documentos de la carpeta `/docs`, los convertirá en vectores usando el modelo de embedding de tu configuración y creará el índice FAISS en `/storage`.

```bash
# Asegúrate de estar en el directorio raíz del proyecto
python rag_flask/ingest.py
```
*Este paso solo necesitas hacerlo una vez, o cada vez que actualices tus documentos.*

#### **Paso 2: Iniciar el Servidor API**
Esto levanta la aplicación web con la que podrás interactuar.

```bash
python rag_flask/app.py
```
El servidor se iniciará en `http://localhost:5000`.

Puedes **sobrescribir** la configuración por defecto al iniciar el servidor:
```bash
# Ejemplo para usar modelos de OpenAI en lugar de los del JSON
python rag_flask/app.py --llm-provider openai --embedding-provider openai
```

#### **Paso 3: Interactuar con el Sistema**

1.  **Interfaz Web (Recomendado)**:
    -   Abre tu navegador y ve a `http://localhost:5000`.
    -   Encontrarás una interfaz sencilla para escribir tus preguntas y ver las respuestas generadas.

2.  **API (para desarrolladores)**:
    Puedes usar `curl` o cualquier cliente de API para interactuar con los endpoints.

    -   **Realizar una consulta:**
        ```bash
        curl -X POST http://localhost:5000/ask \
             -H "Content-Type: application/json" \
             -d '{"message": "¿Qué necesito para obtener un certificado de antecedentes?"}'
        ```

    -   **Verificar la salud del sistema:**
        ```bash
        curl http://localhost:5000/health
        ```

    -   **Ver la configuración actual:**
        ```bash
        curl http://localhost:5000/config
        ```

---

## 📁 5. Estructura del Proyecto

-   `rag_flask/`: Directorio principal de la aplicación.
    -   `app.py`: Servidor web Flask y definición de endpoints.
    -   `rag_chain.py`: Contiene la lógica principal del pipeline RAG.
    -   `ingest.py`: Script para procesar documentos y construir el índice vectorial.
    -   `preprocessing.py`: Funciones de limpieza y preparación de texto.
    -   `model_providers.py`: Abstracción para comunicarse con diferentes APIs de modelos (OpenAI, Ollama, etc.).
    -   `config/rag_config.json`: **Archivo de configuración central.**
    -   `docs/`: Aquí debes colocar los documentos fuente en formato JSON.
    -   `storage/`: Donde se guarda el índice FAISS y los metadatos de validación.

---

## ✅ 6. Conclusión

Esta solución RAG representa un sistema completo, bien fundamentado y listo para producción. No solo implementa la arquitectura básica, sino que incorpora **soluciones explícitas a problemas críticos** como la consistencia de modelos y la optimización de parámetros. La externalización de la configuración lo hace flexible y fácil de mantener, sentando las bases para futuras mejoras como la evaluación continua y el soporte multimodal. 