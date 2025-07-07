#!/bin/bash

# =============================================================================
# RAG Flask - Evaluación Comprehensiva
# =============================================================================
# Script para ejecutar evaluación completa del sistema RAG
# Incluye métricas de recuperación, generación y análisis

set -e

echo "📊 RAG Flask - Evaluación"
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

# Default values
RUN_BASIC=true
RUN_PERFORMANCE=false
RUN_COMPARISON=false
SAVE_REPORTS=true
VERBOSE=false

# Check if virtual environment exists and activate
check_and_activate_venv() {
    if [ ! -d "venv" ]; then
        print_error "Entorno virtual no encontrado. Ejecuta ./setup.sh primero"
        exit 1
    fi
    
    print_info "Activando entorno virtual..."
    source venv/bin/activate
    print_status "Entorno virtual activado"
}

# Check if system is ready for evaluation
check_system_ready() {
    print_info "Verificando que el sistema esté listo..."
    
    # Check if index exists
    if [ ! -f "vector_store/index.faiss" ] || [ ! -f "index_metadata.json" ]; then
        print_error "Índice no encontrado. Indexa documentos primero con:"
        print_info "python ingest.py"
        exit 1
    fi
    
    # Check if evaluation module is available
    if ! python -c "import evaluation" 2>/dev/null; then
        print_error "Módulo de evaluación no encontrado"
        exit 1
    fi
    
    # Check if environment is configured
    if [ -f ".env" ]; then
        source .env
    fi
    
    print_status "Sistema listo para evaluación"
}

# Run basic evaluation
run_basic_evaluation() {
    print_info "🔍 Ejecutando evaluación básica..."
    
    cat > run_basic_eval.py << 'EOF'
import sys
from evaluation import RAGEvaluator

try:
    print("🚀 Iniciando evaluación básica...")
    evaluator = RAGEvaluator()
    
    print("📊 Ejecutando evaluación completa...")
    results = evaluator.evaluate()
    
    print("\n" + "="*50)
    print("📈 RESULTADOS DE EVALUACIÓN")
    print("="*50)
    
    # Retrieval metrics
    if 'retrieval_metrics' in results:
        print("\n🔍 MÉTRICAS DE RECUPERACIÓN:")
        retrieval = results['retrieval_metrics']
        for k in [1, 3, 5]:
            if f'recall_at_{k}' in retrieval:
                print(f"  Recall@{k}: {retrieval[f'recall_at_{k}']:.3f}")
        for k in [1, 3, 5]:
            if f'precision_at_{k}' in retrieval:
                print(f"  Precision@{k}: {retrieval[f'precision_at_{k}']:.3f}")
        if 'mrr' in retrieval:
            print(f"  MRR: {retrieval['mrr']:.3f}")
    
    # Generation metrics
    if 'generation_metrics' in results:
        print("\n🤖 MÉTRICAS DE GENERACIÓN:")
        generation = results['generation_metrics']
        if 'avg_semantic_similarity' in generation:
            print(f"  Similitud Semántica: {generation['avg_semantic_similarity']:.3f}")
        if 'avg_response_time' in generation:
            print(f"  Tiempo de Respuesta: {generation['avg_response_time']:.2f}s")
        if 'avg_response_length' in generation:
            print(f"  Longitud de Respuesta: {generation['avg_response_length']:.0f} chars")
    
    # Chunking metrics
    if 'chunking_metrics' in results:
        print("\n📄 MÉTRICAS DE CHUNKING:")
        chunking = results['chunking_metrics']
        if 'consistency_score' in chunking:
            print(f"  Consistencia: {chunking['consistency_score']:.3f}")
        if 'avg_chunk_size' in chunking:
            print(f"  Tamaño Promedio: {chunking['avg_chunk_size']:.0f} tokens")
        if 'size_std' in chunking:
            print(f"  Desviación Estándar: {chunking['size_std']:.0f} tokens")
    
    # Overall assessment
    print("\n🎯 EVALUACIÓN GENERAL:")
    total_score = 0
    score_count = 0
    
    if 'retrieval_metrics' in results and 'mrr' in results['retrieval_metrics']:
        total_score += results['retrieval_metrics']['mrr']
        score_count += 1
    
    if 'generation_metrics' in results and 'avg_semantic_similarity' in results['generation_metrics']:
        total_score += results['generation_metrics']['avg_semantic_similarity']
        score_count += 1
    
    if score_count > 0:
        overall_score = total_score / score_count
        print(f"  Puntuación General: {overall_score:.3f}")
        if overall_score >= 0.8:
            print("  Estado: ✅ EXCELENTE")
        elif overall_score >= 0.6:
            print("  Estado: ⚡ BUENO")
        elif overall_score >= 0.4:
            print("  Estado: ⚠️  REGULAR")
        else:
            print("  Estado: ❌ NECESITA MEJORAS")
    
    print("\n" + "="*50)
    print("✅ Evaluación básica completada")
    
except Exception as e:
    print(f"❌ Error en evaluación: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    python run_basic_eval.py
    if [ $? -eq 0 ]; then
        print_status "Evaluación básica completada"
    else
        print_error "Error en evaluación básica"
        exit 1
    fi
    
    # Cleanup
    rm -f run_basic_eval.py
}

# Run performance evaluation
run_performance_evaluation() {
    print_info "⚡ Ejecutando evaluación de rendimiento..."
    
    if [ ! -f "performance_test.py" ]; then
        print_warning "performance_test.py no encontrado, saltando evaluación de rendimiento"
        return
    fi
    
    python performance_test.py
    if [ $? -eq 0 ]; then
        print_status "Evaluación de rendimiento completada"
    else
        print_warning "Advertencias en evaluación de rendimiento"
    fi
}

# Run comparison evaluation
run_comparison_evaluation() {
    print_info "🔄 Ejecutando evaluación comparativa..."
    
    cat > run_comparison_eval.py << 'EOF'
import os
import json
from evaluation import RAGEvaluator

def compare_configurations():
    """Compare different chunking configurations"""
    print("🔄 Comparando configuraciones de chunking...")
    
    configurations = [
        {"chunk_size": 256, "chunk_overlap": 26},
        {"chunk_size": 384, "chunk_overlap": 38},
        {"chunk_size": 512, "chunk_overlap": 51},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n🧪 Probando configuración: {config}")
        
        # Temporarily set environment variables
        os.environ['CHUNK_SIZE'] = str(config['chunk_size'])
        os.environ['CHUNK_OVERLAP'] = str(config['chunk_overlap'])
        
        try:
            evaluator = RAGEvaluator()
            result = evaluator.evaluate()
            results[f"config_{config['chunk_size']}_{config['chunk_overlap']}"] = result
            print(f"✅ Configuración {config} evaluada")
        except Exception as e:
            print(f"❌ Error evaluando {config}: {e}")
    
    # Find best configuration
    if results:
        print("\n📊 COMPARACIÓN DE CONFIGURACIONES:")
        best_config = None
        best_score = 0
        
        for config_name, result in results.items():
            score = 0
            count = 0
            
            if 'retrieval_metrics' in result and 'mrr' in result['retrieval_metrics']:
                score += result['retrieval_metrics']['mrr']
                count += 1
            
            if 'generation_metrics' in result and 'avg_semantic_similarity' in result['generation_metrics']:
                score += result['generation_metrics']['avg_semantic_similarity']
                count += 1
            
            if count > 0:
                avg_score = score / count
                print(f"  {config_name}: {avg_score:.3f}")
                if avg_score > best_score:
                    best_score = avg_score
                    best_config = config_name
        
        if best_config:
            print(f"\n🏆 Mejor configuración: {best_config} (score: {best_score:.3f})")

try:
    compare_configurations()
    print("\n✅ Evaluación comparativa completada")
except Exception as e:
    print(f"❌ Error en evaluación comparativa: {e}")
EOF
    
    python run_comparison_eval.py
    if [ $? -eq 0 ]; then
        print_status "Evaluación comparativa completada"
    else
        print_warning "Advertencias en evaluación comparativa"
    fi
    
    # Cleanup
    rm -f run_comparison_eval.py
}

# Generate comprehensive report
generate_report() {
    if [ "$SAVE_REPORTS" = "true" ]; then
        print_info "📝 Generando reporte comprensivo..."
        
        # Create reports directory
        mkdir -p reports
        
        # Generate timestamp
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        REPORT_FILE="reports/evaluation_report_${TIMESTAMP}.md"
        
        cat > "$REPORT_FILE" << EOF
# RAG System - Reporte de Evaluación

**Fecha:** $(date)
**Sistema:** RAG Flask Application

## Configuración del Sistema

- **Proveedor LLM:** ${LLM_PROVIDER:-"No especificado"}
- **Modelo LLM:** ${LLM_MODEL:-"No especificado"}
- **Proveedor Embeddings:** ${EMBEDDING_PROVIDER:-"No especificado"}
- **Modelo Embeddings:** ${EMBEDDING_MODEL:-"No especificado"}
- **Tamaño de Chunk:** ${CHUNK_SIZE:-"No especificado"}
- **Overlap de Chunk:** ${CHUNK_OVERLAP:-"No especificado"}

## Resumen de la Evaluación

Este reporte documenta los resultados de la evaluación comprensiva del sistema RAG.

### Tipos de Evaluación Ejecutados

EOF
        
        if [ "$RUN_BASIC" = "true" ]; then
            echo "- ✅ Evaluación Básica" >> "$REPORT_FILE"
        fi
        
        if [ "$RUN_PERFORMANCE" = "true" ]; then
            echo "- ✅ Evaluación de Rendimiento" >> "$REPORT_FILE"
        fi
        
        if [ "$RUN_COMPARISON" = "true" ]; then
            echo "- ✅ Evaluación Comparativa" >> "$REPORT_FILE"
        fi
        
        cat >> "$REPORT_FILE" << EOF

### Archivos Generados

- Logs de evaluación disponibles en la salida del terminal
- Metadatos del índice: \`index_metadata.json\`
- Configuración del sistema: archivos \`.env\`

### Recomendaciones

1. **Optimización de Chunking**: Basar el tamaño en el modelo de embeddings usado
2. **Monitoreo de Rendimiento**: Revisar tiempos de respuesta regularmente
3. **Actualización de Índice**: Re-indexar cuando se cambien modelos o documentos
4. **Evaluación Periódica**: Ejecutar evaluaciones después de cambios significativos

### Próximos Pasos

- Revisar métricas específicas en la salida detallada
- Ajustar configuración basándose en resultados
- Considerar A/B testing para optimizaciones
- Implementar monitoreo continuo en producción

---
*Reporte generado automáticamente por el sistema de evaluación RAG*
EOF
        
        print_status "Reporte guardado en: $REPORT_FILE"
    fi
}

# Show system information
show_system_info() {
    print_info "🔧 Información del Sistema:"
    
    # Check current configuration
    if [ -f ".env" ]; then
        source .env
    fi
    
    echo "  • Proveedor LLM: ${LLM_PROVIDER:-"No configurado"}"
    echo "  • Modelo LLM: ${LLM_MODEL:-"No configurado"}"
    echo "  • Proveedor Embeddings: ${EMBEDDING_PROVIDER:-"No configurado"}"
    echo "  • Modelo Embeddings: ${EMBEDDING_MODEL:-"No configurado"}"
    
    # Check index information
    if [ -f "index_metadata.json" ]; then
        python -c "
import json
with open('index_metadata.json', 'r') as f:
    metadata = json.load(f)
print(f\"  • Total de chunks: {metadata.get('total_chunks', 'N/A')}\")
print(f\"  • Modelo usado para indexar: {metadata.get('embedding_model', 'N/A')}\")
print(f\"  • Fecha de indexación: {metadata.get('created_at', 'N/A')}\")
" 2>/dev/null || echo "  • Metadatos del índice: No disponibles"
    else
        echo "  • Índice: No encontrado"
    fi
    
    echo ""
}

# Main execution function
main() {
    echo "Iniciando evaluación del sistema RAG..."
    echo ""
    
    show_system_info
    check_and_activate_venv
    check_system_ready
    
    echo ""
    print_info "🚀 Iniciando evaluaciones seleccionadas..."
    echo ""
    
    # Run selected evaluations
    if [ "$RUN_BASIC" = "true" ]; then
        run_basic_evaluation
        echo ""
    fi
    
    if [ "$RUN_PERFORMANCE" = "true" ]; then
        run_performance_evaluation
        echo ""
    fi
    
    if [ "$RUN_COMPARISON" = "true" ]; then
        run_comparison_evaluation
        echo ""
    fi
    
    # Generate report
    generate_report
    
    echo ""
    print_status "🎉 ¡Evaluación completada!"
    
    if [ "$SAVE_REPORTS" = "true" ]; then
        print_info "📁 Reportes guardados en el directorio 'reports/'"
    fi
}

# Show help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "RAG Flask - Evaluación Comprensiva"
    echo ""
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  --help, -h          Mostrar esta ayuda"
    echo "  --basic             Solo evaluación básica (default)"
    echo "  --performance       Incluir evaluación de rendimiento"
    echo "  --comparison        Incluir evaluación comparativa"
    echo "  --all               Ejecutar todas las evaluaciones"
    echo "  --no-reports        No guardar reportes"
    echo "  --verbose           Salida detallada"
    echo ""
    echo "Tipos de evaluación:"
    echo "  • Básica: Métricas de recuperación, generación y chunking"
    echo "  • Rendimiento: Testing de diferentes configuraciones"
    echo "  • Comparativa: Comparación entre configuraciones de chunk"
    echo ""
    echo "Este script:"
    echo "1. Verifica que el sistema esté listo (índice, configuración)"
    echo "2. Ejecuta las evaluaciones seleccionadas"
    echo "3. Genera reportes detallados"
    echo "4. Proporciona recomendaciones de optimización"
    echo ""
    echo "Ejemplos:"
    echo "  $0                    # Evaluación básica"
    echo "  $0 --all              # Todas las evaluaciones"
    echo "  $0 --performance      # Solo rendimiento"
    echo ""
    exit 0
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --basic)
            RUN_BASIC=true
            RUN_PERFORMANCE=false
            RUN_COMPARISON=false
            shift
            ;;
        --performance)
            RUN_PERFORMANCE=true
            shift
            ;;
        --comparison)
            RUN_COMPARISON=true
            shift
            ;;
        --all)
            RUN_BASIC=true
            RUN_PERFORMANCE=true
            RUN_COMPARISON=true
            shift
            ;;
        --no-reports)
            SAVE_REPORTS=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Opción desconocida: $1"
            echo "Usa --help para ver opciones disponibles"
            exit 1
            ;;
    esac
done

# Set verbose output if requested
if [ "$VERBOSE" = "true" ]; then
    set -x
fi

# Run main function
main 