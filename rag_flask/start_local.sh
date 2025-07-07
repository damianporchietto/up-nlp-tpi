#!/bin/bash

# =============================================================================
# RAG Flask - Desarrollo Local con Ollama
# =============================================================================
# Script para desarrollo local usando modelos de Ollama
# No requiere claves de API externas

set -e

echo "🚀 RAG Flask - Desarrollo Local"
echo "==============================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_error "Entorno virtual no encontrado. Ejecuta ./setup.sh primero"
        exit 1
    fi
    print_status "Entorno virtual encontrado"
}

# Activate virtual environment
activate_venv() {
    print_info "Activando entorno virtual..."
    source venv/bin/activate
    print_status "Entorno virtual activado"
}

# Check if Ollama is running
check_ollama() {
    print_info "Verificando Ollama..."
    
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama no está instalado"
        print_info "Instala Ollama desde: https://ollama.ai/"
        exit 1
    fi
    
    # Check if Ollama service is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_warning "Ollama no está ejecutándose"
        print_info "Iniciando Ollama..."
        ollama serve &
        OLLAMA_PID=$!
        echo $OLLAMA_PID > .ollama_pid
        
        # Wait for Ollama to start
        echo -n "Esperando a que Ollama inicie"
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo ""
                print_status "Ollama iniciado correctamente"
                break
            fi
            echo -n "."
            sleep 1
        done
        
        if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_error "No se pudo iniciar Ollama"
            exit 1
        fi
    else
        print_status "Ollama ya está ejecutándose"
    fi
}

# Check and download required models
check_models() {
    print_info "Verificando modelos requeridos..."
    
    # Check embedding model
    if ! ollama list | grep -q "nomic-embed-text"; then
        print_warning "Modelo de embeddings no encontrado. Descargando..."
        ollama pull nomic-embed-text
        print_status "Modelo nomic-embed-text descargado"
    else
        print_status "Modelo de embeddings: nomic-embed-text ✓"
    fi
    
    # Check LLM model
    MODELS=("mistral" "llama3.2:3b" "phi3")
    LLM_MODEL=""
    
    for model in "${MODELS[@]}"; do
        if ollama list | grep -q "$model"; then
            LLM_MODEL="$model"
            print_status "Modelo LLM encontrado: $model ✓"
            break
        fi
    done
    
    if [ -z "$LLM_MODEL" ]; then
        print_warning "No se encontró ningún modelo LLM. Descargando mistral..."
        ollama pull mistral
        LLM_MODEL="mistral"
        print_status "Modelo mistral descargado"
    fi
    
    export LOCAL_LLM_MODEL="$LLM_MODEL"
}

# Configure environment for local development
configure_local_env() {
    print_info "Configurando entorno para desarrollo local..."
    
    # Create or update .env.local
    cat > .env.local << EOF
# Local Development Configuration with Ollama
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000

# Ollama Configuration
LLM_PROVIDER=ollama
LLM_MODEL=${LOCAL_LLM_MODEL}
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Override any other settings for local dev
# CHUNK_SIZE=384
# CHUNK_OVERLAP=38
EOF
    
    # Load local environment
    if [ -f ".env" ]; then
        print_info "Cargando configuración base desde .env"
        source .env
    fi
    
    # Override with local settings
    source .env.local
    print_status "Configuración local aplicada"
}

# Check if documents are indexed
check_index() {
    print_info "Verificando índice de documentos..."
    
    if [ ! -f "vector_store/index.faiss" ] || [ ! -f "index_metadata.json" ]; then
        print_warning "Índice no encontrado. ¿Indexar documentos ahora? (y/N)"
        read -r index_response
        if [[ "$index_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_info "Indexando documentos..."
            python ingest.py
            if [ $? -eq 0 ]; then
                print_status "Documentos indexados correctamente"
            else
                print_error "Error indexando documentos"
                exit 1
            fi
        else
            print_warning "Continuando sin índice. Algunas funciones pueden no funcionar."
        fi
    else
        print_status "Índice de documentos encontrado"
        
        # Validate model consistency
        python -c "
from rag_chain import validate_embedding_consistency
try:
    validate_embedding_consistency()
    print('✅ Consistencia de modelos validada')
except Exception as e:
    print(f'⚠️  Advertencia de consistencia: {e}')
    print('💡 Considera re-indexar con: python ingest.py')
" 2>/dev/null || print_warning "No se pudo validar consistencia de modelos"
    fi
}

# Start the Flask application
start_app() {
    print_info "Iniciando aplicación Flask..."
    print_info "La aplicación estará disponible en: http://localhost:${PORT:-5000}"
    print_info "Presiona Ctrl+C para detener"
    echo ""
    print_info "Endpoints disponibles:"
    echo "  • GET  /health           - Estado del sistema"
    echo "  • GET  /config           - Configuración actual"
    echo "  • GET  /system-info      - Información del sistema"
    echo "  • POST /query            - Consultar documentos"
    echo "  • POST /index            - Re-indexar documentos"
    echo ""
    
    # Start Flask with local configuration
    export FLASK_APP=app.py
    export FLASK_ENV=development
    
    python app.py
}

# Cleanup function
cleanup() {
    print_info "Limpiando recursos..."
    
    # Kill Ollama if we started it
    if [ -f ".ollama_pid" ]; then
        OLLAMA_PID=$(cat .ollama_pid)
        if ps -p $OLLAMA_PID > /dev/null 2>&1; then
            print_info "Deteniendo Ollama (PID: $OLLAMA_PID)..."
            kill $OLLAMA_PID 2>/dev/null || true
        fi
        rm -f .ollama_pid
    fi
    
    print_status "Limpieza completada"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    echo "Iniciando desarrollo local con Ollama..."
    echo ""
    
    check_venv
    activate_venv
    check_ollama
    check_models
    configure_local_env
    check_index
    
    echo ""
    print_status "¡Configuración lista!"
    echo ""
    
    start_app
}

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "RAG Flask - Desarrollo Local"
    echo ""
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  --help, -h     Mostrar esta ayuda"
    echo "  --no-index     No verificar/crear índice de documentos"
    echo "  --port PORT    Puerto para la aplicación (default: 5000)"
    echo ""
    echo "Este script:"
    echo "1. Verifica que Ollama esté instalado y ejecutándose"
    echo "2. Descarga modelos necesarios si no existen"
    echo "3. Configura el entorno para desarrollo local"
    echo "4. Verifica/crea el índice de documentos"
    echo "5. Inicia la aplicación Flask"
    echo ""
    echo "Modelos usados:"
    echo "  • Embeddings: nomic-embed-text"
    echo "  • LLM: mistral (o el primer modelo compatible encontrado)"
    echo ""
    exit 0
fi

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-index)
            export SKIP_INDEX_CHECK=1
            shift
            ;;
        --port)
            export PORT="$2"
            shift 2
            ;;
        *)
            print_error "Opción desconocida: $1"
            echo "Usa --help para ver opciones disponibles"
            exit 1
            ;;
    esac
done

# Run main function
main 