#!/bin/bash

# =============================================================================
# RAG Flask - Utilidades de Desarrollo
# =============================================================================
# Script con utilidades comunes para desarrollo y testing

set -e

echo "🛠️  RAG Flask - Dev Utils"
echo "========================"

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

# Check virtual environment
check_venv() {
    if [ ! -d "venv" ]; then
        print_error "Entorno virtual no encontrado. Ejecuta ./setup.sh primero"
        exit 1
    fi
    source venv/bin/activate
}

# Test API endpoints
test_api() {
    print_info "🔍 Testeando endpoints de la API..."
    
    # Check if server is running
    if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
        print_error "Servidor no está ejecutándose en puerto 5000"
        print_info "Inicia el servidor con ./start_local.sh o ./start_production.sh"
        exit 1
    fi
    
    echo ""
    print_info "📡 Probando endpoints:"
    
    # Test health endpoint
    echo -n "  • /health: "
    if curl -s http://localhost:5000/health | grep -q "healthy"; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ ERROR${NC}"
    fi
    
    # Test config endpoint
    echo -n "  • /config: "
    if curl -s http://localhost:5000/config > /dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ ERROR${NC}"
    fi
    
    # Test system-info endpoint
    echo -n "  • /system-info: "
    if curl -s http://localhost:5000/system-info > /dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ ERROR${NC}"
    fi
    
    # Test query endpoint with sample query
    echo -n "  • /query (POST): "
    QUERY_RESULT=$(curl -s -X POST http://localhost:5000/query \
        -H "Content-Type: application/json" \
        -d '{"query": "test query", "max_results": 1}' 2>/dev/null)
    
    if echo "$QUERY_RESULT" | grep -q "answer\|documents"; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ ERROR${NC}"
    fi
    
    echo ""
    print_status "Testing de API completado"
}

# Debug system status
debug_system() {
    print_info "🔧 Debug del sistema..."
    
    echo ""
    print_info "📊 Estado del sistema:"
    
    # Check Python environment
    echo "  • Python: $(python --version 2>&1)"
    echo "  • Virtual env: ${VIRTUAL_ENV:-"No activo"}"
    
    # Check required modules
    echo ""
    print_info "📦 Módulos principales:"
    
    for module in flask langchain faiss openai; do
        echo -n "  • $module: "
        if python -c "import $module; print('✅ OK')" 2>/dev/null; then
            true  # Already printed by Python
        else
            echo -e "${RED}❌ FALTA${NC}"
        fi
    done
    
    # Check project modules
    echo ""
    print_info "🏗️  Módulos del proyecto:"
    
    for module in preprocessing model_providers rag_chain evaluation; do
        echo -n "  • $module: "
        if python -c "import $module; print('✅ OK')" 2>/dev/null; then
            true
        else
            echo -e "${RED}❌ ERROR${NC}"
        fi
    done
    
    # Check configuration
    echo ""
    print_info "⚙️  Configuración:"
    
    if [ -f ".env" ]; then
        source .env
        echo "  • Archivo .env: ✅ Encontrado"
        echo "  • LLM Provider: ${LLM_PROVIDER:-"No configurado"}"
        echo "  • Embedding Provider: ${EMBEDDING_PROVIDER:-"No configurado"}"
        if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "sk-your-openai-api-key-here" ]; then
            echo "  • OpenAI API Key: ✅ Configurada"
        else
            echo "  • OpenAI API Key: ❌ No configurada"
        fi
    else
        echo "  • Archivo .env: ❌ No encontrado"
    fi
    
    # Check index
    echo ""
    print_info "🗂️  Índice vectorial:"
    
    if [ -f "vector_store/index.faiss" ]; then
        echo "  • Archivo índice: ✅ Encontrado"
        echo "  • Tamaño: $(du -sh vector_store/index.faiss | cut -f1)"
    else
        echo "  • Archivo índice: ❌ No encontrado"
    fi
    
    if [ -f "index_metadata.json" ]; then
        echo "  • Metadatos: ✅ Encontrados"
        python -c "
import json
with open('index_metadata.json', 'r') as f:
    meta = json.load(f)
print(f'  • Total chunks: {meta.get(\"total_chunks\", \"N/A\")}')
print(f'  • Modelo usado: {meta.get(\"embedding_model\", \"N/A\")}')
" 2>/dev/null || echo "  • Error leyendo metadatos"
    else
        echo "  • Metadatos: ❌ No encontrados"
    fi
}

# Quick test with sample query
quick_test() {
    print_info "⚡ Test rápido del sistema..."
    
    check_venv
    
    # Test model loading
    python -c "
from model_providers import get_embeddings_model, get_llm_model
from rag_chain import create_rag_chain

print('🔍 Probando modelo de embeddings...')
embed_model = get_embeddings_model()
test_embedding = embed_model.embed_query('test query')
print(f'✅ Embedding generado: {len(test_embedding)} dimensiones')

print('🤖 Probando modelo LLM...')
llm_model = get_llm_model()
print('✅ Modelo LLM cargado')

print('🔗 Probando cadena RAG...')
rag_chain = create_rag_chain()
print('✅ Cadena RAG creada')

print('📝 Realizando consulta de prueba...')
try:
    result = rag_chain.invoke({
        'question': '¿Qué es un certificado de antecedentes?'
    })
    print(f'✅ Consulta exitosa: {len(result.get(\"answer\", \"\"))} caracteres de respuesta')
except Exception as e:
    print(f'⚠️  Advertencia en consulta: {e}')
    
print('🎉 Test rápido completado')
"
    
    if [ $? -eq 0 ]; then
        print_status "Test rápido exitoso"
    else
        print_error "Errores en test rápido"
        exit 1
    fi
}

# Benchmark performance
benchmark() {
    print_info "📊 Benchmark de rendimiento..."
    
    check_venv
    
    cat > benchmark_test.py << 'EOF'
import time
import statistics
from model_providers import get_embeddings_model, get_llm_model
from rag_chain import create_rag_chain

def benchmark_embeddings(model, queries, iterations=3):
    """Benchmark embedding generation"""
    times = []
    
    for query in queries:
        query_times = []
        for _ in range(iterations):
            start = time.time()
            model.embed_query(query)
            end = time.time()
            query_times.append(end - start)
        times.extend(query_times)
    
    return {
        'avg': statistics.mean(times),
        'min': min(times),
        'max': max(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0
    }

def benchmark_rag_chain(chain, queries, iterations=2):
    """Benchmark RAG chain"""
    times = []
    
    for query in queries:
        for _ in range(iterations):
            start = time.time()
            try:
                chain.invoke({'question': query})
                end = time.time()
                times.append(end - start)
            except Exception as e:
                print(f'Error en query "{query}": {e}')
    
    return {
        'avg': statistics.mean(times) if times else 0,
        'min': min(times) if times else 0,
        'max': max(times) if times else 0,
        'std': statistics.stdev(times) if len(times) > 1 else 0
    }

# Test queries
test_queries = [
    "¿Qué documentos necesito?",
    "¿Cuáles son los requisitos?",
    "¿Cómo solicito un turno?",
    "¿Dónde puedo hacer el trámite?"
]

print("🚀 Iniciando benchmark...")

# Benchmark embeddings
print("🔍 Benchmark de embeddings...")
embed_model = get_embeddings_model()
embed_results = benchmark_embeddings(embed_model, test_queries)

print(f"  • Tiempo promedio: {embed_results['avg']:.3f}s")
print(f"  • Tiempo mínimo: {embed_results['min']:.3f}s")
print(f"  • Tiempo máximo: {embed_results['max']:.3f}s")
print(f"  • Desviación estándar: {embed_results['std']:.3f}s")

# Benchmark RAG chain
print("\n🔗 Benchmark de cadena RAG...")
rag_chain = create_rag_chain()
rag_results = benchmark_rag_chain(rag_chain, test_queries)

print(f"  • Tiempo promedio: {rag_results['avg']:.3f}s")
print(f"  • Tiempo mínimo: {rag_results['min']:.3f}s")
print(f"  • Tiempo máximo: {rag_results['max']:.3f}s")
print(f"  • Desviación estándar: {rag_results['std']:.3f}s")

print("\n📊 Resumen de rendimiento:")
print(f"  • Embeddings por segundo: {1/embed_results['avg']:.1f}")
print(f"  • Consultas RAG por minuto: {60/rag_results['avg']:.1f}" if rag_results['avg'] > 0 else "  • Consultas RAG: Error")

print("\n✅ Benchmark completado")
EOF
    
    python benchmark_test.py
    rm -f benchmark_test.py
    
    print_status "Benchmark completado"
}

# Show logs
show_logs() {
    print_info "📋 Mostrando logs del sistema..."
    
    if [ -d "logs" ]; then
        echo ""
        print_info "📁 Archivos de log disponibles:"
        ls -la logs/ 2>/dev/null || echo "No hay archivos de log"
        
        if [ -f "logs/app.log" ]; then
            echo ""
            print_info "🔍 Últimas 20 líneas del log principal:"
            tail -20 logs/app.log
        fi
        
        if [ -f "logs/error.log" ]; then
            echo ""
            print_info "❌ Últimas 10 líneas del log de errores:"
            tail -10 logs/error.log
        fi
    else
        print_warning "Directorio logs no encontrado"
    fi
}

# Index documents
reindex() {
    print_info "🗂️  Re-indexando documentos..."
    
    check_venv
    
    if [ -f "vector_store/index.faiss" ]; then
        print_warning "⚠️  Esto eliminará el índice actual"
        echo -n "¿Continuar? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_info "Re-indexación cancelada"
            return
        fi
    fi
    
    python ingest.py
    
    if [ $? -eq 0 ]; then
        print_status "Re-indexación completada"
    else
        print_error "Error en re-indexación"
        exit 1
    fi
}

# Show help
show_help() {
    echo "RAG Flask - Utilidades de Desarrollo"
    echo ""
    echo "Uso: $0 <comando> [opciones]"
    echo ""
    echo "Comandos disponibles:"
    echo "  test-api           Probar endpoints de la API"
    echo "  debug              Mostrar estado del sistema"
    echo "  quick-test         Test rápido de funcionalidad"
    echo "  benchmark          Benchmark de rendimiento"
    echo "  logs               Mostrar logs del sistema"
    echo "  reindex            Re-indexar documentos"
    echo "  help               Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  $0 debug           # Ver estado del sistema"
    echo "  $0 test-api        # Probar que la API funcione"
    echo "  $0 quick-test      # Test rápido completo"
    echo "  $0 benchmark       # Medir rendimiento"
    echo ""
}

# Main function
main() {
    case "${1:-help}" in
        "test-api")
            test_api
            ;;
        "debug")
            check_venv
            debug_system
            ;;
        "quick-test")
            quick_test
            ;;
        "benchmark")
            benchmark
            ;;
        "logs")
            show_logs
            ;;
        "reindex")
            reindex
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Comando desconocido: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 