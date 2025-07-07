#!/bin/bash

# =============================================================================
# RAG Flask - Producción con OpenAI
# =============================================================================
# Script para despliegue en producción usando modelos de OpenAI
# Requiere OPENAI_API_KEY configurada

set -e

echo "🚀 RAG Flask - Producción"
echo "========================="

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

# Check OpenAI API key
check_openai_key() {
    print_info "Verificando configuración de OpenAI..."
    
    if [ -z "$OPENAI_API_KEY" ]; then
        if [ -f ".env" ]; then
            source .env
        fi
        
        if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-your-openai-api-key-here" ]; then
            print_error "OPENAI_API_KEY no configurada o es un placeholder"
            print_info "Configura tu clave de API en el archivo .env:"
            print_info "OPENAI_API_KEY=sk-tu-clave-real-aqui"
            exit 1
        fi
    fi
    
    # Test API key validity
    python -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
try:
    client = openai.OpenAI()
    models = client.models.list()
    print('✅ Clave de API de OpenAI válida')
except Exception as e:
    print(f'❌ Error con la clave de API: {e}')
    exit(1)
" 2>/dev/null || {
        print_error "Error validando clave de OpenAI"
        exit 1
    }
    
    print_status "Configuración de OpenAI verificada"
}

# Configure production environment
configure_production_env() {
    print_info "Configurando entorno de producción..."
    
    # Create production environment file
    cat > .env.production << EOF
# Production Configuration with OpenAI
FLASK_ENV=production
FLASK_DEBUG=False
PORT=${PORT:-5000}

# OpenAI Configuration
LLM_PROVIDER=openai
LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-3-large}

# Performance optimizations for production
CHUNK_SIZE=${CHUNK_SIZE:-512}
CHUNK_OVERLAP=${CHUNK_OVERLAP:-51}

# Security and logging
LOG_LEVEL=${LOG_LEVEL:-INFO}
MAX_WORKERS=${MAX_WORKERS:-4}
EOF
    
    # Load base environment
    if [ -f ".env" ]; then
        print_info "Cargando configuración base desde .env"
        source .env
    fi
    
    # Apply production overrides
    source .env.production
    print_status "Configuración de producción aplicada"
    
    # Set additional production variables
    export FLASK_ENV=production
    export FLASK_DEBUG=False
}

# Validate production requirements
validate_production() {
    print_info "Validando requisitos de producción..."
    
    # Check required packages
    python -c "
import sys
required_packages = ['flask', 'langchain', 'faiss-cpu', 'openai', 'gunicorn']
missing = []

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        missing.append(package)

if missing:
    print(f'❌ Paquetes faltantes: {missing}')
    print('💡 Instala con: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('✅ Todos los paquetes requeridos están instalados')
"
    
    # Check if gunicorn is available for production
    if ! python -c "import gunicorn" 2>/dev/null; then
        print_warning "Gunicorn no encontrado. Instalando..."
        pip install gunicorn
        print_status "Gunicorn instalado"
    fi
    
    print_status "Validación de producción completada"
}

# Check and prepare index
prepare_index() {
    print_info "Preparando índice de documentos..."
    
    if [ ! -f "vector_store/index.faiss" ] || [ ! -f "index_metadata.json" ]; then
        print_warning "Índice no encontrado. Creando índice para producción..."
        print_info "Esto puede tardar varios minutos con modelos de OpenAI..."
        
        python ingest.py
        if [ $? -eq 0 ]; then
            print_status "Índice creado correctamente"
        else
            print_error "Error creando índice"
            exit 1
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
    print(f'⚠️  Advertencia: {e}')
    print('💡 Considera re-indexar para producción')
" 2>/dev/null
    fi
    
    # Check index quality
    python -c "
import os
import json
if os.path.exists('index_metadata.json'):
    with open('index_metadata.json', 'r') as f:
        metadata = json.load(f)
    chunk_count = metadata.get('total_chunks', 0)
    print(f'📊 Índice contiene {chunk_count} chunks')
    if chunk_count < 10:
        print('⚠️  Pocos chunks en el índice. Considera agregar más documentos.')
" 2>/dev/null
}

# Health check before starting
health_check() {
    print_info "Realizando verificación de salud..."
    
    # Test model initialization
    python -c "
from model_providers import get_embeddings_model, get_llm_model
from rag_chain import create_rag_chain

try:
    print('🔍 Probando modelo de embeddings...')
    embed_model = get_embeddings_model()
    test_embedding = embed_model.embed_query('test')
    print(f'✅ Embeddings: {len(test_embedding)} dimensiones')
    
    print('🤖 Probando modelo LLM...')
    llm_model = get_llm_model()
    print('✅ Modelo LLM inicializado')
    
    print('🔗 Probando cadena RAG...')
    rag_chain = create_rag_chain()
    print('✅ Cadena RAG creada correctamente')
    
    print('✅ Verificación de salud completada')
except Exception as e:
    print(f'❌ Error en verificación: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "Verificación de salud exitosa"
    else
        print_error "Fallo en verificación de salud"
        exit 1
    fi
}

# Start production server
start_production_server() {
    print_info "Iniciando servidor de producción..."
    
    # Configuration
    WORKERS=${MAX_WORKERS:-4}
    PORT=${PORT:-5000}
    TIMEOUT=${TIMEOUT:-120}
    
    print_info "Configuración del servidor:"
    echo "  • Puerto: $PORT"
    echo "  • Workers: $WORKERS"
    echo "  • Timeout: $TIMEOUT segundos"
    echo "  • Modelo LLM: ${LLM_MODEL:-gpt-4o-mini}"
    echo "  • Modelo Embeddings: ${EMBEDDING_MODEL:-text-embedding-3-large}"
    echo ""
    
    print_info "La aplicación estará disponible en: http://localhost:$PORT"
    print_info "Endpoints de producción:"
    echo "  • GET  /health           - Estado del sistema"
    echo "  • GET  /config           - Configuración (sin claves sensibles)"
    echo "  • POST /query            - Consultar documentos"
    echo "  • GET  /metrics          - Métricas del sistema (si está habilitado)"
    echo ""
    print_info "Para monitoreo: tail -f logs/app.log"
    echo ""
    
    # Create logs directory
    mkdir -p logs
    
    # Start with gunicorn for production
    exec gunicorn \
        --bind 0.0.0.0:$PORT \
        --workers $WORKERS \
        --timeout $TIMEOUT \
        --worker-class sync \
        --worker-connections 1000 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --preload \
        --access-logfile logs/access.log \
        --error-logfile logs/error.log \
        --log-level info \
        --capture-output \
        app:app
}

# Start with Flask for simpler production
start_flask_production() {
    print_info "Iniciando servidor Flask en modo producción..."
    
    PORT=${PORT:-5000}
    print_info "La aplicación estará disponible en: http://localhost:$PORT"
    echo ""
    
    # Create logs directory
    mkdir -p logs
    
    # Start Flask app
    python app.py
}

# Main execution
main() {
    echo "Iniciando despliegue de producción..."
    echo ""
    
    check_venv
    activate_venv
    check_openai_key
    configure_production_env
    validate_production
    prepare_index
    health_check
    
    echo ""
    print_status "¡Configuración de producción lista!"
    echo ""
    
    # Choose server type
    if command -v gunicorn &> /dev/null && [ "$USE_GUNICORN" != "false" ]; then
        start_production_server
    else
        if [ "$USE_GUNICORN" = "false" ]; then
            print_info "Gunicorn deshabilitado por configuración"
        else
            print_warning "Gunicorn no disponible, usando Flask"
        fi
        start_flask_production
    fi
}

# Show help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "RAG Flask - Producción"
    echo ""
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  --help, -h         Mostrar esta ayuda"
    echo "  --port PORT        Puerto del servidor (default: 5000)"
    echo "  --workers N        Número de workers para gunicorn (default: 4)"
    echo "  --no-gunicorn      Usar Flask en lugar de gunicorn"
    echo "  --model MODEL      Modelo LLM de OpenAI (default: gpt-4o-mini)"
    echo ""
    echo "Variables de entorno importantes:"
    echo "  OPENAI_API_KEY     Clave de API de OpenAI (requerida)"
    echo "  LLM_MODEL          Modelo LLM (gpt-4o-mini, gpt-4, etc.)"
    echo "  EMBEDDING_MODEL    Modelo de embeddings (text-embedding-3-large)"
    echo "  MAX_WORKERS        Número de workers"
    echo "  PORT               Puerto del servidor"
    echo ""
    echo "Este script:"
    echo "1. Valida la configuración de OpenAI"
    echo "2. Configura el entorno de producción"
    echo "3. Verifica todos los requisitos"
    echo "4. Prepara/valida el índice de documentos"
    echo "5. Ejecuta verificaciones de salud"
    echo "6. Inicia el servidor con gunicorn (recomendado) o Flask"
    echo ""
    exit 0
fi

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            export PORT="$2"
            shift 2
            ;;
        --workers)
            export MAX_WORKERS="$2"
            shift 2
            ;;
        --no-gunicorn)
            export USE_GUNICORN=false
            shift
            ;;
        --model)
            export LLM_MODEL="$2"
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