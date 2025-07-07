#!/bin/bash

# =============================================================================
# RAG Flask - Limpieza y Mantenimiento
# =============================================================================
# Script para limpiar recursos temporales y realizar mantenimiento del sistema

set -e

echo "🧹 RAG Flask - Limpieza y Mantenimiento"
echo "======================================="

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

# Default values
CLEAN_CACHE=false
CLEAN_LOGS=false
CLEAN_INDEX=false
CLEAN_REPORTS=false
CLEAN_VENV=false
CLEAN_ALL=false
DRY_RUN=false
CONFIRM=true

# Function to get directory size
get_dir_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1 || echo "N/A"
    else
        echo "N/A"
    fi
}

# Function to get file count in directory
get_file_count() {
    if [ -d "$1" ]; then
        find "$1" -type f 2>/dev/null | wc -l || echo "0"
    else
        echo "0"
    fi
}

# Show current disk usage
show_disk_usage() {
    print_info "📊 Uso actual de espacio en disco:"
    echo ""
    
    # Check various directories and files
    echo "  📁 Entorno virtual (venv/):"
    echo "     Tamaño: $(get_dir_size "venv")"
    echo "     Archivos: $(get_file_count "venv")"
    echo ""
    
    echo "  🗂️  Índice vectorial (vector_store/):"
    echo "     Tamaño: $(get_dir_size "vector_store")"
    echo "     Archivos: $(get_file_count "vector_store")"
    echo ""
    
    echo "  📋 Logs (logs/):"
    echo "     Tamaño: $(get_dir_size "logs")"
    echo "     Archivos: $(get_file_count "logs")"
    echo ""
    
    echo "  📊 Reportes (reports/):"
    echo "     Tamaño: $(get_dir_size "reports")"
    echo "     Archivos: $(get_file_count "reports")"
    echo ""
    
    echo "  🗄️  Cache de Python (__pycache__):"
    PYCACHE_SIZE=$(find . -name "__pycache__" -type d -exec du -sh {} + 2>/dev/null | awk '{total+=$1} END {print total "M"}' || echo "0M")
    PYCACHE_COUNT=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l || echo "0")
    echo "     Tamaño: ${PYCACHE_SIZE}"
    echo "     Directorios: ${PYCACHE_COUNT}"
    echo ""
    
    # Check for temporary files
    TEMP_FILES=$(find . -name "*.tmp" -o -name "*.temp" -o -name "*.log.*" -o -name "core.*" 2>/dev/null | wc -l || echo "0")
    if [ "$TEMP_FILES" -gt 0 ]; then
        echo "  🗑️  Archivos temporales encontrados: $TEMP_FILES"
        echo ""
    fi
}

# Clean Python cache
clean_python_cache() {
    print_info "🗄️  Limpiando cache de Python..."
    
    if [ "$DRY_RUN" = "true" ]; then
        find . -name "__pycache__" -type d 2>/dev/null | head -10
        print_info "DRY RUN: Se eliminarían $(find . -name "__pycache__" -type d 2>/dev/null | wc -l) directorios de cache"
        return
    fi
    
    # Remove __pycache__ directories
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove .pyc files
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove .pyo files
    find . -name "*.pyo" -delete 2>/dev/null || true
    
    print_status "Cache de Python limpiado"
}

# Clean logs
clean_logs() {
    print_info "📋 Limpiando archivos de log..."
    
    if [ ! -d "logs" ]; then
        print_info "Directorio logs no existe"
        return
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        find logs -name "*.log" -o -name "*.log.*" 2>/dev/null | head -10
        print_info "DRY RUN: Se eliminarían $(find logs -name "*.log" -o -name "*.log.*" 2>/dev/null | wc -l) archivos de log"
        return
    fi
    
    # Keep only the most recent log files (last 5)
    if [ -f "logs/app.log" ]; then
        mv logs/app.log logs/app.log.backup
    fi
    
    # Remove old log files
    find logs -name "*.log" -delete 2>/dev/null || true
    find logs -name "*.log.*" -delete 2>/dev/null || true
    
    # Restore current log
    if [ -f "logs/app.log.backup" ]; then
        mv logs/app.log.backup logs/app.log
    fi
    
    print_status "Archivos de log limpiados"
}

# Clean vector index
clean_vector_index() {
    print_warning "🗂️  ¡ADVERTENCIA! Esto eliminará el índice vectorial completo"
    
    if [ "$CONFIRM" = "true" ] && [ "$DRY_RUN" = "false" ]; then
        print_warning "Tendrás que re-indexar todos los documentos después"
        echo -n "¿Estás seguro de que quieres continuar? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_info "Limpieza de índice cancelada"
            return
        fi
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        print_info "DRY RUN: Se eliminarían:"
        [ -d "vector_store" ] && echo "  - vector_store/"
        [ -f "index_metadata.json" ] && echo "  - index_metadata.json"
        return
    fi
    
    # Remove vector store directory
    if [ -d "vector_store" ]; then
        rm -rf vector_store
        print_status "Directorio vector_store eliminado"
    fi
    
    # Remove index metadata
    if [ -f "index_metadata.json" ]; then
        rm -f index_metadata.json
        print_status "Metadatos del índice eliminados"
    fi
    
    print_warning "⚠️  Recuerda re-indexar con: python ingest.py"
}

# Clean reports
clean_reports() {
    print_info "📊 Limpiando reportes antiguos..."
    
    if [ ! -d "reports" ]; then
        print_info "Directorio reports no existe"
        return
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        find reports -name "*.md" -o -name "*.html" -o -name "*.json" 2>/dev/null | head -10
        print_info "DRY RUN: Se eliminarían $(find reports -name "*.md" -o -name "*.html" -o -name "*.json" 2>/dev/null | wc -l) archivos de reporte"
        return
    fi
    
    # Keep only the most recent 3 reports
    find reports -name "evaluation_report_*.md" -type f | sort -r | tail -n +4 | xargs rm -f 2>/dev/null || true
    find reports -name "performance_report_*.json" -type f | sort -r | tail -n +4 | xargs rm -f 2>/dev/null || true
    
    print_status "Reportes antiguos limpiados (mantenidos los 3 más recientes)"
}

# Clean virtual environment
clean_virtual_env() {
    print_warning "🐍 ¡ADVERTENCIA! Esto eliminará el entorno virtual completo"
    
    if [ "$CONFIRM" = "true" ] && [ "$DRY_RUN" = "false" ]; then
        print_warning "Tendrás que ejecutar ./setup.sh nuevamente después"
        echo -n "¿Estás seguro de que quieres continuar? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_info "Limpieza de entorno virtual cancelada"
            return
        fi
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        [ -d "venv" ] && print_info "DRY RUN: Se eliminaría el directorio venv/ ($(get_dir_size "venv"))"
        return
    fi
    
    if [ -d "venv" ]; then
        rm -rf venv
        print_status "Entorno virtual eliminado"
        print_info "💡 Ejecuta ./setup.sh para recrear el entorno"
    fi
}

# Clean temporary files
clean_temp_files() {
    print_info "🗑️  Limpiando archivos temporales..."
    
    if [ "$DRY_RUN" = "true" ]; then
        find . -name "*.tmp" -o -name "*.temp" -o -name ".*.swp" -o -name "*~" 2>/dev/null | head -10
        print_info "DRY RUN: Se eliminarían $(find . -name "*.tmp" -o -name "*.temp" -o -name ".*.swp" -o -name "*~" 2>/dev/null | wc -l) archivos temporales"
        return
    fi
    
    # Remove common temporary files
    find . -name "*.tmp" -delete 2>/dev/null || true
    find . -name "*.temp" -delete 2>/dev/null || true
    find . -name ".*.swp" -delete 2>/dev/null || true
    find . -name "*~" -delete 2>/dev/null || true
    find . -name "core.*" -delete 2>/dev/null || true
    
    # Remove specific temporary files from the project
    rm -f .env.local .env.production 2>/dev/null || true
    rm -f .ollama_pid 2>/dev/null || true
    rm -f run_*.py 2>/dev/null || true
    
    print_status "Archivos temporales eliminados"
}

# Optimize and maintenance
run_optimization() {
    print_info "⚡ Ejecutando optimización y mantenimiento..."
    
    # Activate virtual environment if exists
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || true
    fi
    
    # Update pip and dependencies if in venv
    if [ -n "$VIRTUAL_ENV" ]; then
        print_info "📦 Actualizando pip..."
        if [ "$DRY_RUN" = "false" ]; then
            python -m pip install --upgrade pip --quiet 2>/dev/null || true
            print_status "Pip actualizado"
        else
            print_info "DRY RUN: Se actualizaría pip"
        fi
    fi
    
    # Check for broken symlinks
    BROKEN_LINKS=$(find . -type l ! -exec test -e {} \; -print 2>/dev/null | wc -l || echo "0")
    if [ "$BROKEN_LINKS" -gt 0 ]; then
        if [ "$DRY_RUN" = "false" ]; then
            find . -type l ! -exec test -e {} \; -delete 2>/dev/null || true
            print_status "Enlaces simbólicos rotos eliminados: $BROKEN_LINKS"
        else
            print_info "DRY RUN: Se eliminarían $BROKEN_LINKS enlaces simbólicos rotos"
        fi
    fi
}

# Main cleanup function
run_cleanup() {
    echo "Iniciando proceso de limpieza..."
    echo ""
    
    show_disk_usage
    
    if [ "$CLEAN_ALL" = "true" ]; then
        CLEAN_CACHE=true
        CLEAN_LOGS=true
        CLEAN_REPORTS=true
        # Note: Not including index and venv in "all" for safety
    fi
    
    # Always clean temporary files and Python cache
    clean_temp_files
    clean_python_cache
    
    if [ "$CLEAN_LOGS" = "true" ]; then
        clean_logs
    fi
    
    if [ "$CLEAN_REPORTS" = "true" ]; then
        clean_reports
    fi
    
    if [ "$CLEAN_INDEX" = "true" ]; then
        clean_vector_index
    fi
    
    if [ "$CLEAN_VENV" = "true" ]; then
        clean_virtual_env
    fi
    
    # Run optimization
    run_optimization
    
    echo ""
    print_status "🎉 Limpieza completada"
    
    # Show final disk usage
    echo ""
    print_info "📊 Uso de espacio después de la limpieza:"
    show_disk_usage
}

# Show help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "RAG Flask - Limpieza y Mantenimiento"
    echo ""
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones de limpieza:"
    echo "  --cache             Limpiar cache de Python (__pycache__, *.pyc)"
    echo "  --logs              Limpiar archivos de log antiguos"
    echo "  --reports           Limpiar reportes antiguos (mantiene los 3 más recientes)"
    echo "  --index             ⚠️  Eliminar índice vectorial (requiere re-indexación)"
    echo "  --venv              ⚠️  Eliminar entorno virtual (requiere ./setup.sh)"
    echo "  --all               Limpiar cache, logs y reportes (sin index/venv)"
    echo ""
    echo "Opciones de control:"
    echo "  --dry-run           Mostrar qué se eliminaría sin hacerlo"
    echo "  --no-confirm        No pedir confirmación para operaciones peligrosas"
    echo "  --help, -h          Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  $0                  # Limpieza básica (temp + cache)"
    echo "  $0 --all            # Limpieza completa segura"
    echo "  $0 --dry-run --all  # Ver qué se eliminaría"
    echo "  $0 --index          # Eliminar índice (requiere confirmación)"
    echo ""
    echo "Operaciones siempre ejecutadas:"
    echo "  • Limpieza de archivos temporales"
    echo "  • Limpieza de cache de Python"
    echo "  • Optimización básica del sistema"
    echo ""
    echo "⚠️  Las opciones --index y --venv requieren confirmación (usa --no-confirm para automatizar)"
    echo ""
    exit 0
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cache)
            CLEAN_CACHE=true
            shift
            ;;
        --logs)
            CLEAN_LOGS=true
            shift
            ;;
        --reports)
            CLEAN_REPORTS=true
            shift
            ;;
        --index)
            CLEAN_INDEX=true
            shift
            ;;
        --venv)
            CLEAN_VENV=true
            shift
            ;;
        --all)
            CLEAN_ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-confirm)
            CONFIRM=false
            shift
            ;;
        *)
            print_error "Opción desconocida: $1"
            echo "Usa --help para ver opciones disponibles"
            exit 1
            ;;
    esac
done

# Show dry run notice
if [ "$DRY_RUN" = "true" ]; then
    print_warning "🔍 MODO DRY RUN: Solo mostrando qué se eliminaría"
    echo ""
fi

# Run cleanup
run_cleanup 