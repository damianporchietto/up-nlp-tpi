#!/bin/bash
# Script para probar múltiples configuraciones de modelos

set -e  # Salir si hay error

PORT=5000
API_URL="http://localhost:$PORT"
TEST_DIR="test"
RESULTS_DIR="$TEST_DIR/results"

# Función para limpiar procesos al salir
cleanup() {
    echo "Limpiando procesos..."
    # Matar cualquier proceso de Python que esté usando el puerto
    pkill -f "python app.py" || true
    # Matar cualquier proceso de Python que esté ejecutando test_questions.py
    pkill -f "python test/test_questions.py" || true
    echo "Limpieza completada"
}

# Registrar la función de limpieza para que se ejecute al salir
trap cleanup EXIT

# Crear directorio de resultados si no existe
mkdir -p "$RESULTS_DIR"

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
  
  # Ejecutar pruebas usando test_questions.py
  echo "Ejecutando pruebas..."
  RESULTS_FILE="$RESULTS_DIR/${model_name}_results.json"
  python "$TEST_DIR/test_questions.py" --output "$RESULTS_FILE" --api-url "$API_URL"
  
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
  # Using a smaller, open model that doesn't require authentication
  run_test "huggingface_opt" "huggingface" "facebook/opt-1.3b" "huggingface" "BAAI/bge-small-en-v1.5"
else
  echo "Omitiendo pruebas de HuggingFace (dependencias no instaladas)"
fi

echo ""
echo "=========================================="
echo "TODAS LAS PRUEBAS COMPLETADAS"
echo "Los resultados se han guardado en $RESULTS_DIR/"
echo "==========================================" 