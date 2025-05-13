# Asistente de Trámites de Córdoba - RAG API

Servicio de Generación Aumentada por Recuperación (RAG) para consultas sobre trámites y servicios gubernamentales de la Provincia de Córdoba. Construido con **Flask** y **LangChain**.

## Estructura del proyecto

```
rag_flask/
├── app.py               # API Flask (GET /, GET /health, GET /config, GET /providers, POST /ask)
├── rag_chain.py         # Pipeline RAG (embeddings + vector DB + LLM)
├── ingest.py            # Creación de base de datos vectorial a partir de archivos JSON en ./docs
├── model_providers.py   # Módulo para gestión de diferentes proveedores de modelos (OpenAI, Ollama, etc.)
├── setup_env.sh         # Script para configurar entorno virtual e instalar dependencias
├── test_models.py       # Script para probar modelos individuales
├── test_multiple_models.sh # Script para probar múltiples configuraciones de modelos
├── requirements.txt     # Dependencias Python
├── docs/                # Documentos fuente (archivos JSON organizados por ministerio)
│   ├── MINISTERIO DE GOBIERNO/
│   ├── SECRETARÍA DE AMBIENTE/
│   └── ...
├── storage/             # Índice FAISS persistente (generado automáticamente)
└── .env.example         # Variables de entorno
```

## Configuración rápida con script

La forma más fácil de comenzar es utilizando el script de configuración automática:

```bash
# Dar permisos de ejecución al script (si es necesario)
chmod +x setup_env.sh

# Ejecutar el script de configuración
./setup_env.sh
```

El script realizará automáticamente las siguientes tareas:
1. Crear un entorno virtual Python en la carpeta `venv/`
2. Instalar todas las dependencias básicas
3. Ofrecer la opción de instalar dependencias para proveedores adicionales (Ollama, HuggingFace)
4. Crear un archivo `.env` con la configuración predeterminada

## Inicio rápido (manual)

Si prefieres configurar el entorno manualmente:

```bash
# 1. Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Agregar clave de OpenAI (si usas OpenAI como proveedor)
cp .env.example .env
echo 'OPENAI_API_KEY=sk‑...' >> .env

# 4. Preparar la base de conocimientos
python ingest.py          # Lee los archivos JSON en ./docs y construye el índice FAISS

# 5. Ejecutar la API
python app.py
```

## Usando diferentes proveedores de modelos

Este proyecto soporta múltiples proveedores de modelos de lenguaje (LLM) y embeddings:

### Proveedores disponibles:
- **OpenAI**: Modelos GPT y embeddings (requiere API key)
- **Ollama**: Modelos locales como Llama, Mistral (requiere instalar Ollama)
- **HuggingFace**: Modelos locales o mediante API

### Configuración mediante variables de entorno

Puedes configurar los proveedores en el archivo `.env`:

```
OPENAI_API_KEY=sk-...        # Requerido para usar OpenAI

# Configuración de proveedores
LLM_PROVIDER=openai          # openai, ollama, huggingface 
LLM_MODEL=gpt-4o-mini        # Específico para cada proveedor
EMBEDDING_PROVIDER=openai    # openai, ollama, huggingface
EMBEDDING_MODEL=text-embedding-3-large
```

### Configuración mediante línea de comandos

Al iniciar el servidor puedes especificar los proveedores:

```bash
# Ejemplo con OpenAI
python app.py --llm-provider openai --llm-model gpt-4o-mini

# Ejemplo con Ollama (requiere tener Ollama instalado)
python app.py --llm-provider ollama --llm-model mistral --embedding-provider ollama --embedding-model nomic-embed-text
```

### Instalación de dependencias para diferentes proveedores

Para utilizar proveedores distintos a OpenAI, descomenta e instala las dependencias necesarias en `requirements.txt`:

```bash
# Para HuggingFace
pip install langchain-huggingface transformers torch sentence-transformers accelerate
```

## Consultar la API

La API se inicia en http://localhost:5000 con la siguiente documentación:

- **GET /** - Documentación de la API
- **GET /health** - Verificar estado del servicio
- **GET /config** - Ver configuración actual de modelos
- **GET /providers** - Listar proveedores disponibles
- **POST /ask** - Realizar consultas sobre trámites

### Ejemplo de consulta:

```bash
curl -X POST http://localhost:5000/ask \
     -H 'Content-Type: application/json' \
     -d '{"message": "¿Qué necesito para obtener un certificado de antecedentes?"}'
```

### Respuesta:

```json
{
  "answer": "Para obtener un certificado de antecedentes (Certificado de Buena Conducta) en la Provincia de Córdoba necesitas:\n\n1. Tener domicilio en Córdoba\n2. DNI en perfecto estado\n3. Comprobante de pago de la Tasa Retributiva de Servicio (original)\n4. DNI original\n\nEl trámite es personal e intransferible, tiene costo y requiere turno previo. Puedes realizarlo en la Policía de la Provincia de Córdoba. El vencimiento del certificado lo determina la entidad, organismo o institución que lo requiera.\n\nPuedes consultar el estado del trámite online en: https://sistemas.policiacordoba.gov.ar/consultacertificado/\n\nPara más información, visita: https://www.policiacordoba.gov.ar/tramites-y-servicios/certificado-de-antecedentes/",
  "sources": [
    {
      "source": "/home/user/rag_flask/docs/MINISTERIO DE GOBIERNO/275ead14-aeb0-ee11-baa9-005056a1885b.json",
      "title": "CERTIFICADO DE ANTECEDENTES PARA ARGENTINOS NATIVOS O POR OPCIÓN",
      "url": "https://cidi.cba.gov.ar/portal-publico/tramite/275ead14-aeb0-ee11-baa9-005056a1885b",
      "snippet": "Title: CERTIFICADO DE ANTECEDENTES PARA ARGENTINOS NATIVOS O POR OPCIÓN\n\nDescription: Solicitud de constancia que certifique que una persona registra o no antecedentes penales y/o contravencionales en la jurisdicción de la Provincia de Córdoba..."
    }
  ]
}
```

## Pruebas de rendimiento

El proyecto incluye herramientas para probar y comparar diferentes configuraciones de modelos:

```bash
# Probar una única configuración
python test_models.py --model-name "nombre_prueba"

# Probar múltiples configuraciones de modelos automáticamente
./test_multiple_models.sh
```

## Personalización

* Añade más archivos JSON de documentos gubernamentales en `docs/` siguiendo la estructura existente y vuelve a ejecutar `python ingest.py`.
* Puedes modificar el prompt en `rag_chain.py` para ajustar la forma en que se procesan las consultas.
* Para añadir historial conversacional, puedes extender el sistema usando `ConversationalRetrievalChain` de LangChain.

## Recreación del índice con diferentes embeddings

Si deseas cambiar el proveedor de embeddings, necesitarás reconstruir el índice FAISS:

```bash
# Reconstruir el índice con un proveedor y modelo específico
python ingest.py --provider ollama --model nomic-embed-text
```

## Dependencias principales

- Flask: Framework web ligero
- LangChain: Framework para aplicaciones basadas en LLM
- FAISS: Biblioteca de búsqueda de similitud y agrupación de vectores
- OpenAI/Ollama/HuggingFace: Proveedores de modelos de lenguaje y embeddings