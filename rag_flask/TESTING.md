# Guía de Testing - Asistente de Trámites de Córdoba

Este documento proporciona instrucciones para probar y utilizar el sistema RAG implementado para consultas sobre trámites y servicios gubernamentales de la Provincia de Córdoba.

## Preparación automática (Recomendado)

Para configurar el entorno de manera rápida y sencilla, utiliza el script de configuración incluido:

```bash
# Dar permisos de ejecución (si es necesario)
chmod +x setup_env.sh

# Ejecutar script de configuración
./setup_env.sh
```

Este script:
1. Crea un entorno virtual Python
2. Instala todas las dependencias necesarias
3. Te permite elegir qué proveedores de modelos deseas utilizar
4. Configura un archivo `.env` con valores predeterminados

## Preparación manual

Si prefieres configurar el entorno manualmente:

1. Asegúrate de haber instalado todas las dependencias:
   ```bash
   # Crear entorno virtual
   python3 -m venv venv
   source venv/bin/activate
   
   # Instalar dependencias
   pip install -r requirements.txt
   ```

2. Crea un archivo `.env` con tu clave de API de OpenAI:
   ```bash
   echo 'OPENAI_API_KEY=sk-your-api-key-here' > .env
   ```

3. Construye la base de datos vectorial:
   ```bash
   python ingest.py
   ```
   Esto procesará todos los archivos JSON dentro del directorio `docs/` y creará un índice FAISS en la carpeta `storage/`.

## Iniciar el servidor

Ejecuta la aplicación Flask:
```bash
python app.py
```

Por defecto, el servidor se iniciará en `http://localhost:5000`.

## Pruebas básicas

### 1. Verificar el estado del servicio

```bash
curl http://localhost:5000/health
```

Deberías recibir una respuesta como:
```json
{"status": "ok"}
```

### 2. Consultar la documentación

Abre en tu navegador: `http://localhost:5000/`

Deberías ver la página de documentación HTML que describe los endpoints disponibles.

### 3. Realizar consultas sobre trámites

Puedes probar el sistema con diferentes consultas relacionadas con trámites gubernamentales:

#### Ejemplo 1: Certificado de antecedentes

```bash
curl -X POST http://localhost:5000/ask \
     -H 'Content-Type: application/json' \
     -d '{"message": "¿Qué necesito para obtener un certificado de antecedentes?"}'
```

#### Ejemplo 2: Requisitos específicos

```bash
curl -X POST http://localhost:5000/ask \
     -H 'Content-Type: application/json' \
     -d '{"message": "¿Dónde puedo tramitar un certificado de antecedentes?"}'
```

#### Ejemplo 3: Consulta sobre costos

```bash
curl -X POST http://localhost:5000/ask \
     -H 'Content-Type: application/json' \
     -d '{"message": "¿Tiene costo obtener un certificado de antecedentes?"}'
```

## Pruebas con diferentes proveedores de modelos

El sistema ahora permite probar diferentes proveedores de modelos de lenguaje (LLM) y embeddings.

### 1. Consultar proveedores disponibles

```bash
curl http://localhost:5000/providers
```

Deberías recibir información sobre los proveedores disponibles.

### 2. Consultar la configuración actual

```bash
curl http://localhost:5000/config
```

### 3. Pruebas con OpenAI (configuración por defecto)

```bash
# Iniciar el servidor con OpenAI (por defecto)
python app.py

# Realizar consultas normalmente
curl -X POST http://localhost:5000/ask \
     -H 'Content-Type: application/json' \
     -d '{"message": "¿Qué documentos necesito para un certificado de antecedentes?"}'
```

### 4. Pruebas con Ollama (modelos locales)

Primero, asegúrate de tener Ollama instalado y funcionando:
```bash
# Instalar Ollama según las instrucciones en https://ollama.ai/

# Descargar modelos necesarios
ollama pull mistral
ollama pull nomic-embed-text

# Instalar dependencias necesarias
pip install langchain_community
```

Luego, ejecuta el servidor con Ollama como proveedor:
```bash
python app.py --llm-provider ollama --llm-model mistral --embedding-provider ollama --embedding-model nomic-embed-text
```

Si necesitas recrear el índice con embeddings de Ollama:
```bash
python ingest.py --provider ollama --model nomic-embed-text
```

### 5. Pruebas con HuggingFace (modelos locales o API)

Primero, instala las dependencias necesarias:
```bash
pip install langchain-huggingface transformers torch sentence-transformers accelerate
```

Luego, ejecuta el servidor con HuggingFace como proveedor:
```bash
python app.py --llm-provider huggingface --llm-model google/flan-t5-base --embedding-provider huggingface --embedding-model BAAI/bge-small-en-v1.5
```

Si necesitas recrear el índice con embeddings de HuggingFace:
```bash
python ingest.py --provider huggingface --model BAAI/bge-small-en-v1.5
```

## Comparación de resultados entre diferentes modelos

Para evaluar y comparar el rendimiento de diferentes modelos, puedes:

1. Preparar un conjunto de preguntas de prueba en un archivo (por ejemplo, `test_questions.txt`).
2. Crear un script que envíe cada pregunta al servidor y guarde las respuestas.
3. Ejecutar el script con diferentes configuraciones de modelos.
4. Comparar las respuestas para cada pregunta.

Ejemplo de script de prueba (save as `test_models.py`):

```python
import requests
import json
import time
from pathlib import Path

# Cargar preguntas de prueba
with open('test_questions.txt', 'r') as f:
    questions = [line.strip() for line in f if line.strip()]

# Configurar modelo a probar
model_name = "openai"  # Cambiar según el modelo a probar

# Carpeta para resultados
results_dir = Path(f"results_{model_name}")
results_dir.mkdir(exist_ok=True)

# Probar cada pregunta
for i, question in enumerate(questions):
    print(f"Testing question {i+1}/{len(questions)}: {question[:50]}...")
    
    try:
        response = requests.post(
            "http://localhost:5000/ask",
            json={"message": question},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Guardar resultado
            with open(results_dir / f"q{i+1}.json", 'w') as f:
                json.dump({
                    "question": question,
                    "answer": result.get("answer"),
                    "sources": result.get("sources", [])
                }, f, indent=2)
                
            print(f"  ✓ Success")
        else:
            print(f"  ✗ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")
    
    # Esperar para no sobrecargar la API
    time.sleep(1)

print("Testing complete!")
```

## Automatizando pruebas de múltiples modelos

Para facilitar las pruebas con diferentes configuraciones, puedes usar el script `test_multiple_models.sh`:

```bash
# Ejecuta pruebas automáticas con distintas configuraciones de modelos
./test_multiple_models.sh
```

Este script:
1. Crea un archivo de preguntas de prueba si no existe
2. Verifica las dependencias y disponibilidad de modelos
3. Prueba automáticamente diferentes configuraciones de modelos
4. Guarda los resultados para su comparación

## Evaluación del sistema

Para evaluar la calidad de las respuestas, considera los siguientes aspectos:

1. **Precisión**: ¿La respuesta contiene información correcta basada en los documentos de origen?
2. **Relevancia**: ¿La respuesta aborda directamente la consulta del usuario?
3. **Fuentes**: ¿Se citan correctamente las fuentes de la información?
4. **Completitud**: ¿La respuesta proporciona toda la información necesaria?
5. **Claridad**: ¿La respuesta es fácil de entender?
6. **Velocidad**: ¿Cuánto tiempo tarda cada modelo en responder?
7. **Robustez**: ¿El modelo maneja bien consultas ambiguas o mal formuladas?

## Posibles mejoras

Si deseas expandir el sistema, considera:

1. Añadir más documentos a la base de conocimientos.
2. Implementar un historial de conversación para mantener el contexto entre consultas.
3. Mejorar el prompt en `rag_chain.py` para obtener respuestas más específicas.
4. Añadir filtrado de documentos por relevancia o categoría.
5. Implementar un frontend web para facilitar las consultas.
6. Añadir soporte para más proveedores de modelos.
7. Implementar mecanismos de caché para mejorar el rendimiento.

## Solución de problemas

### Error al cargar el modelo de embeddings

Si recibes un error relacionado con la API de OpenAI, verifica que:
- La clave API en tu archivo `.env` sea válida
- Tengas saldo suficiente en tu cuenta de OpenAI
- Tu conexión a internet funcione correctamente

### Error con Ollama

Si tienes problemas al usar Ollama:
- Verifica que Ollama esté instalado y en ejecución
- Comprueba que hayas descargado los modelos necesarios
- Asegúrate de que `langchain_community` esté instalado

### Error con HuggingFace

Si tienes problemas al usar modelos de HuggingFace:
- Verifica que todas las dependencias estén instaladas
- Asegúrate de tener suficiente RAM y espacio en disco
- Considera usar modelos más pequeños si tienes limitaciones de recursos

### Error al procesar los documentos JSON

Si encuentras problemas al procesar los documentos:
- Verifica que todos los archivos JSON tengan el formato correcto
- Asegúrate de que la ruta a la carpeta `docs/` sea accesible
- Comprueba que tengas permisos de escritura para crear la carpeta `storage/`

### El servidor no inicia

Si el servidor Flask no inicia:
- Verifica que el puerto 5000 no esté en uso
- Asegúrate de que todas las dependencias estén instaladas correctamente
- Comprueba los registros de error para identificar problemas específicos 