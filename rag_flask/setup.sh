#!/bin/bash

# =============================================================================
# RAG Flask - Setup Script
# =============================================================================
# Script de configuración inicial para el proyecto RAG
# Instala dependencias, configura entorno y valida la instalación

set -e  # Exit on any error

echo "🚀 RAG Flask - Setup Inicial"
echo "============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python 3 is installed
check_python() {
    print_info "Verificando Python..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_status "Python encontrado: $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Versión de Python compatible (≥3.8)"
        else
            print_error "Se requiere Python 3.8 o superior. Versión actual: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 no encontrado. Instala Python 3.8 o superior."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_info "Configurando entorno virtual..."
    
    if [ -d "venv" ]; then
        print_warning "Entorno virtual ya existe. ¿Deseas recrearlo? (y/N)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            rm -rf venv
            print_status "Entorno virtual anterior eliminado"
        else
            print_info "Usando entorno virtual existente"
            return 0
        fi
    fi
    
    python3 -m venv venv
    print_status "Entorno virtual creado"
}

# Activate virtual environment
activate_venv() {
    print_info "Activando entorno virtual..."
    source venv/bin/activate
    print_status "Entorno virtual activado"
}

# Install requirements
install_requirements() {
    print_info "Instalando dependencias básicas..."
    
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install basic requirements
    pip install -r requirements.txt
    print_status "Dependencias básicas instaladas"
    
    # Check for optional providers
    echo ""
    print_info "¿Deseas instalar dependencias para proveedores adicionales?"
    
    # HuggingFace
    print_warning "¿Instalar dependencias para HuggingFace? (y/N)"
    read -r hf_response
    if [[ "$hf_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Instalando dependencias de HuggingFace..."
        pip install transformers torch sentence-transformers accelerate
        print_status "Dependencias de HuggingFace instaladas"
    fi
    
    # Evaluation dependencies
    print_warning "¿Instalar dependencias para evaluación avanzada? (y/N)"
    read -r eval_response
    if [[ "$eval_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Instalando dependencias de evaluación..."
        pip install scikit-learn pandas numpy matplotlib seaborn
        print_status "Dependencias de evaluación instaladas"
    fi
}

# Configure environment variables
configure_env() {
    print_info "Configurando variables de entorno..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_status "Archivo .env creado desde .env.example"
        else
            # Create basic .env file
            cat > .env << EOF
# OpenAI Configuration (for production)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Model Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000

# Optional: Ollama Configuration (for local development)
# OLLAMA_BASE_URL=http://localhost:11434
EOF
            print_status "Archivo .env básico creado"
        fi
        
        print_warning "⚠️  IMPORTANTE: Edita el archivo .env con tus claves de API"
        print_info "Para OpenAI: Agrega tu OPENAI_API_KEY"
        print_info "Para desarrollo local: Considera usar Ollama (ver start_local.sh)"
    else
        print_status "Archivo .env ya existe"
    fi
}

# Download NLTK data if needed
setup_nltk() {
    print_info "Configurando datos de NLTK..."
    python -c "
import nltk
try:
    nltk.data.find('corpora/stopwords')
    print('✅ Datos de NLTK ya disponibles')
except LookupError:
    print('⬇️  Descargando datos de NLTK...')
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print('✅ Datos de NLTK descargados')
" 2>/dev/null || print_warning "Error configurando NLTK (opcional)"
}

# Check if Ollama is available
check_ollama() {
    print_info "Verificando disponibilidad de Ollama..."
    if command -v ollama &> /dev/null; then
        print_status "Ollama encontrado: $(ollama --version 2>/dev/null || echo 'versión desconocida')"
        
        print_warning "¿Deseas descargar modelos de Ollama recomendados? (y/N)"
        read -r ollama_response
        if [[ "$ollama_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_info "Descargando modelos de Ollama..."
            ollama pull nomic-embed-text || print_warning "Error descargando nomic-embed-text"
            ollama pull mistral || print_warning "Error descargando mistral"
            print_status "Modelos de Ollama configurados"
        fi
    else
        print_warning "Ollama no encontrado"
        print_info "Para desarrollo local, considera instalar Ollama:"
        print_info "https://ollama.ai/"
    fi
}

# Validate installation
validate_installation() {
    print_info "Validando instalación..."
    
    # Test imports
    python -c "
import sys
try:
    import flask
    import langchain
    import faiss
    import openai
    print('✅ Dependencias principales: OK')
except ImportError as e:
    print(f'❌ Error importando dependencias: {e}')
    sys.exit(1)

try:
    from preprocessing import normalize_text
    from model_providers import get_embeddings_model
    print('✅ Módulos del proyecto: OK')
except ImportError as e:
    print(f'❌ Error importando módulos del proyecto: {e}')
    sys.exit(1)
    
print('✅ Instalación validada correctamente')
"
    
    if [ $? -eq 0 ]; then
        print_status "Validación de instalación exitosa"
    else
        print_error "Error en validación de instalación"
        exit 1
    fi
}

# Create sample documents if docs directory is empty
setup_sample_docs() {
    print_info "Verificando documentos de ejemplo..."
    
    if [ ! -d "docs" ] || [ -z "$(ls -A docs 2>/dev/null)" ]; then
        print_warning "No se encontraron documentos. ¿Crear ejemplos? (y/N)"
        read -r docs_response
        if [[ "$docs_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            mkdir -p docs/EJEMPLO
            
            cat > docs/EJEMPLO/certificado_antecedentes.json << 'EOF'
{
  "title": "Certificado de Antecedentes",
  "description": "Solicitud de constancia que certifique que una persona registra o no antecedentes penales y/o contravencionales en la jurisdicción de la Provincia de Córdoba.",
  "requirements": [
    {
      "title": "Documentación requerida",
      "content": "DNI en perfecto estado, comprobante de pago de la Tasa Retributiva de Servicio"
    },
    {
      "title": "Condiciones",
      "content": "Tener domicilio en Córdoba, trámite personal e intransferible"
    }
  ],
  "steps": [
    "Solicitar turno previo",
    "Presentarse en Policía de la Provincia de Córdoba",
    "Presentar documentación requerida",
    "Realizar pago de tasas"
  ],
  "url": "https://ejemplo.cba.gov.ar/certificado-antecedentes"
}
EOF
            
            print_status "Documento de ejemplo creado en docs/EJEMPLO/"
        fi
    else
        print_status "Documentos encontrados en directorio docs/"
    fi
}

# Main setup function
main() {
    echo "Iniciando configuración del proyecto RAG Flask..."
    echo ""
    
    check_python
    create_venv
    activate_venv
    install_requirements
    configure_env
    setup_nltk
    check_ollama
    setup_sample_docs
    validate_installation
    
    echo ""
    echo "🎉 ¡Configuración completada exitosamente!"
    echo ""
    print_info "Próximos pasos:"
    echo "1. Edita el archivo .env con tus claves de API"
    echo "2. Para desarrollo local: ./start_local.sh"
    echo "3. Para producción: ./start_production.sh" 
    echo "4. Para evaluación: ./run_evaluation.sh"
    echo ""
    print_info "Para activar el entorno virtual manualmente:"
    echo "source venv/bin/activate"
    echo ""
    print_status "¡El proyecto está listo para usar!"
}

# Run main function
main "$@" 