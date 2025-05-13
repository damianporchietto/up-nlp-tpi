#!/bin/bash
# Script para probar múltiples configuraciones de modelos

set -e  # Salir si hay error

PORT=5000
API_URL="http://localhost:$PORT"
QUESTIONS_FILE="test_questions.txt"

# Si no existe el archivo de preguntas, lo creamos con preguntas por defecto
if [ ! -f "$QUESTIONS_FILE" ]; then
  echo "Creando archivo de preguntas de ejemplo..."
  cat > "$QUESTIONS_FILE" << EOF
¿Qué necesito para obtener un certificado de antecedentes?
¿Dónde puedo tramitar un certificado de antecedentes?
¿Tiene costo obtener un certificado de antecedentes?
¿Necesito sacar turno para tramitar un certificado de antecedentes?
¿Puedo enviar a otra persona a tramitar mi certificado de antecedentes?
EOF
  echo "Archivo $QUESTIONS_FILE creado."
fi

# Función para ejecutar pruebas
run_test() {
  local model_name=$1
  local llm_provider=$2
  local llm_model=$3
  local embedding_provider=$4
  local embedding_model=$5
  
  echo ""
  echo "====================================="
  echo "PROBANDO CONFIGURACIÓN: $model_name"
  echo "====================================="
  echo "LLM Provider: $llm_provider"
  echo "LLM Model: ${llm_model:-default}"
  echo "Embedding Provider: $embedding_provider"
  echo "Embedding Model: ${embedding_model:-default}"
  echo "====================================="
  
  # Iniciar el servidor con la configuración específica
  echo "Iniciando servidor..."
  python app.py --llm-provider "$llm_provider" \
                --llm-model "$llm_model" \
                --embedding-provider "$embedding_provider" \
                --embedding-model "$embedding_model" \
                --port "$PORT" &
  
  SERVER_PID=$!
  
  # Esperar a que el servidor esté listo
  echo "Esperando a que el servidor esté listo..."
  sleep 5
  
  # Ejecutar pruebas
  echo "Ejecutando pruebas..."
  python test_models.py --model-name "$model_name" --questions-file "$QUESTIONS_FILE" --api-url "$API_URL"
  
  # Detener el servidor
  echo "Deteniendo servidor (PID: $SERVER_PID)..."
  kill $SERVER_PID
  wait $SERVER_PID 2>/dev/null || true
  sleep 2
}

# Asegurarse de que exista el entorno virtual y las dependencias instaladas
check_dependencies() {
  case "$1" in
    "openai")
      if [ -z "$OPENAI_API_KEY" ]; then
        echo "ADVERTENCIA: No se ha configurado OPENAI_API_KEY en el entorno"
      fi
      ;;
    "ollama")
      if ! command -v ollama &> /dev/null; then
        echo "ADVERTENCIA: Ollama no parece estar instalado"
      else
        echo "Verificando modelos de Ollama..."
        if ! ollama list | grep -q "$2"; then
          echo "El modelo '$2' no está disponible en Ollama. Puedes descargarlo con: ollama pull $2"
        fi
      fi
      ;;
    "huggingface")
      echo "Verificando dependencias para HuggingFace..."
      if ! pip list | grep -q "transformers"; then
        echo "ADVERTENCIA: El paquete 'transformers' no está instalado"
        echo "Para instalar: pip install transformers torch sentence-transformers"
      fi
      ;;
  esac
}

# Configuraciones a probar
echo "Preparando pruebas de modelos múltiples..."

# Tests de OpenAI (si hay API key)
if [ -n "$OPENAI_API_KEY" ]; then
  check_dependencies "openai"
  run_test "openai_gpt4o_mini" "openai" "gpt-4o-mini" "openai" "text-embedding-3-large"
else
  echo "Omitiendo pruebas de OpenAI (no se encontró OPENAI_API_KEY)"
fi

# Tests de Ollama (si está instalado)
if command -v ollama &> /dev/null; then
  check_dependencies "ollama" "mistral"
  run_test "ollama_mistral" "ollama" "mistral" "ollama" "nomic-embed-text"
else
  echo "Omitiendo pruebas de Ollama (no está instalado)"
fi

# Tests de HuggingFace (si están instaladas las dependencias)
if pip list | grep -q "transformers"; then
  check_dependencies "huggingface"
  run_test "huggingface_flan_t5" "huggingface" "google/flan-t5-base" "huggingface" "BAAI/bge-small-en-v1.5"
else
  echo "Omitiendo pruebas de HuggingFace (dependencias no instaladas)"
fi

echo ""
echo "=========================================="
echo "TODAS LAS PRUEBAS COMPLETADAS"
echo "Revisa las carpetas 'results_*' para ver los resultados de cada prueba."
echo "==========================================" 