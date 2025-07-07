#!/bin/bash

# =============================================================================
# RAG Flask - Quick Start
# =============================================================================
# Script de inicio rápido para nuevos usuarios
# Combina setup inicial y arranque del sistema

set -e

echo "🚀 RAG Flask - Quick Start"
echo "=========================="

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

# Default mode
MODE="local"  # local or production

# Check if this is the first run
check_first_run() {
    if [ ! -d "venv" ] || [ ! -f ".env" ]; then
        print_info "🎯 Primera ejecución detectada"
        return 0
    else
        print_info "🔄 Sistema ya configurado"
        return 1
    fi
}

# Interactive mode selection
choose_mode() {
    print_info "¿Cómo quieres ejecutar RAG Flask?"
    echo ""
    echo "1) 🏠 Desarrollo Local (Ollama) - Gratis, no requiere API keys"
    echo "2) ☁️  Producción (OpenAI) - Requiere OPENAI_API_KEY"
    echo "3) ⚙️  Solo configurar (no iniciar)"
    echo ""
    echo -n "Selecciona una opción (1-3) [1]: "
    read -r choice
    
    case $choice in
        2)
            MODE="production"
            ;;
        3)
            MODE="setup"
            ;;
        *)
            MODE="local"
            ;;
    esac
    
    print_status "Modo seleccionado: $MODE"
}

# Quick setup for first-time users
quick_setup() {
    print_info "🛠️  Ejecutando configuración inicial..."
    
    if [ -f "./setup.sh" ]; then
        # Run setup with automatic yes responses for basic options
        export DEBIAN_FRONTEND=noninteractive
        echo -e "n\nn\n" | ./setup.sh || {
            print_error "Error en configuración inicial"
            exit 1
        }
    else
        print_error "Script setup.sh no encontrado"
        exit 1
    fi
    
    print_status "Configuración inicial completada"
}

# Configure for selected mode
configure_mode() {
    print_info "⚙️  Configurando para modo: $MODE"
    
    case $MODE in
        "local")
            # Check if Ollama is available
            if ! command -v ollama &> /dev/null; then
                print_warning "Ollama no está instalado"
                print_info "📥 Descargando e instalando Ollama..."
                
                # Download and install Ollama
                curl -fsSL https://ollama.ai/install.sh | sh || {
                    print_error "Error instalando Ollama"
                    print_info "Instala manualmente desde: https://ollama.ai/"
                    exit 1
                }
                
                print_status "Ollama instalado correctamente"
            fi
            ;;
            
        "production")
            # Check if OpenAI API key is configured
            if [ -f ".env" ]; then
                source .env
            fi
            
            if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-your-openai-api-key-here" ]; then
                print_warning "OPENAI_API_KEY no configurada"
                print_info "Necesitas configurar tu clave de API de OpenAI"
                echo ""
                echo -n "Ingresa tu OPENAI_API_KEY: "
                read -r api_key
                
                if [ -n "$api_key" ]; then
                    # Update .env file
                    if [ -f ".env" ]; then
                        sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env
                    else
                        echo "OPENAI_API_KEY=$api_key" > .env
                    fi
                    print_status "Clave de API configurada"
                else
                    print_error "Clave de API requerida para modo producción"
                    exit 1
                fi
            fi
            ;;
    esac
}

# Start the appropriate service
start_service() {
    print_info "🚀 Iniciando servicio en modo: $MODE"
    
    case $MODE in
        "local")
            if [ -f "./start_local.sh" ]; then
                exec ./start_local.sh
            else
                print_error "Script start_local.sh no encontrado"
                exit 1
            fi
            ;;
            
        "production")
            if [ -f "./start_production.sh" ]; then
                exec ./start_production.sh
            else
                print_error "Script start_production.sh no encontrado"
                exit 1
            fi
            ;;
            
        "setup")
            print_status "✅ Configuración completada"
            print_info "Para iniciar el sistema:"
            echo "  • Desarrollo local: ./start_local.sh"
            echo "  • Producción: ./start_production.sh"
            echo "  • Evaluación: ./run_evaluation.sh"
            ;;
    esac
}

# Show welcome message
show_welcome() {
    echo ""
    print_info "🎉 ¡Bienvenido a RAG Flask!"
    echo ""
    echo "Este script te ayudará a configurar y ejecutar el sistema RAG."
    echo ""
    print_info "Características del sistema:"
    echo "  • 🤖 RAG con embeddings y LLM"
    echo "  • 📚 Procesamiento de documentos JSON"
    echo "  • 🔍 Búsqueda semántica avanzada"
    echo "  • 📊 Métricas de evaluación"
    echo "  • 🏠 Desarrollo local con Ollama"
    echo "  • ☁️  Producción con OpenAI"
    echo ""
}

# Main function
main() {
    show_welcome
    
    # Check if first run and setup if needed
    if check_first_run; then
        print_info "🛠️  Primera configuración necesaria"
        quick_setup
    fi
    
    # Let user choose mode
    choose_mode
    
    # Configure for selected mode
    configure_mode
    
    # Start service unless setup-only mode
    if [ "$MODE" != "setup" ]; then
        echo ""
        print_info "🎯 Todo listo! Iniciando en 3 segundos..."
        sleep 3
        start_service
    fi
}

# Show help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "RAG Flask - Quick Start"
    echo ""
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  --local             Iniciar directamente en modo local"
    echo "  --production        Iniciar directamente en modo producción"
    echo "  --setup-only        Solo configurar, no iniciar"
    echo "  --help, -h          Mostrar esta ayuda"
    echo ""
    echo "Este script:"
    echo "1. Detecta si es la primera ejecución"
    echo "2. Ejecuta la configuración inicial si es necesario"
    echo "3. Te permite elegir el modo de ejecución"
    echo "4. Configura el entorno apropiado"
    echo "5. Inicia el sistema automáticamente"
    echo ""
    echo "Modos disponibles:"
    echo "  • Local: Usa Ollama (gratis, sin API keys)"
    echo "  • Producción: Usa OpenAI (requiere API key)"
    echo "  • Solo setup: Configura pero no inicia"
    echo ""
    echo "Para usuarios nuevos: simplemente ejecuta ./quick_start.sh"
    echo ""
    exit 0
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            MODE="local"
            shift
            ;;
        --production)
            MODE="production"
            shift
            ;;
        --setup-only)
            MODE="setup"
            shift
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