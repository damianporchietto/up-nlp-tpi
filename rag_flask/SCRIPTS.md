# Scripts de RAG Flask

Esta documentación describe todos los scripts disponibles para el proyecto RAG Flask.

## 🚀 Scripts de Inicio

### `quick_start.sh`
**Script de inicio rápido para nuevos usuarios**
```bash
./quick_start.sh
```
- Detecta primera ejecución y configura automáticamente
- Permite elegir modo (local/producción)
- Instala Ollama si es necesario
- Configura claves de API interactivamente
- Inicia el sistema automáticamente

**Opciones:**
- `--local`: Iniciar directamente en modo local
- `--production`: Iniciar directamente en modo producción
- `--setup-only`: Solo configurar, no iniciar

### `setup.sh`
**Configuración inicial completa del proyecto**
```bash
./setup.sh
```
- Verifica e instala Python 3.8+
- Crea entorno virtual
- Instala dependencias básicas y opcionales
- Configura archivos de entorno
- Descarga datos de NLTK
- Valida instalación completa
- Configura documentos de ejemplo

### `start_local.sh`
**Desarrollo local con Ollama (gratuito)**
```bash
./start_local.sh
```
- Usa modelos locales de Ollama
- No requiere claves de API
- Inicia/verifica Ollama automáticamente
- Descarga modelos necesarios
- Configura entorno de desarrollo
- Valida consistencia del índice

**Opciones:**
- `--no-index`: No verificar/crear índice
- `--port PORT`: Puerto personalizado

### `start_production.sh`
**Producción con OpenAI**
```bash
./start_production.sh
```
- Usa modelos de OpenAI (requiere API key)
- Valida configuración de producción
- Inicia con Gunicorn para mejor rendimiento
- Configuración optimizada para producción
- Logging avanzado y métricas

**Opciones:**
- `--port PORT`: Puerto del servidor
- `--workers N`: Número de workers
- `--no-gunicorn`: Usar Flask simple
- `--model MODEL`: Modelo LLM específico

## 📊 Scripts de Evaluación

### `run_evaluation.sh`
**Evaluación comprensiva del sistema**
```bash
./run_evaluation.sh
```
- Métricas de recuperación (Recall@K, Precision@K, MRR)
- Métricas de generación (similitud semántica, tiempo)
- Análisis de chunking y consistencia
- Reportes detallados con recomendaciones

**Opciones:**
- `--basic`: Solo evaluación básica (default)
- `--performance`: Incluir evaluación de rendimiento
- `--comparison`: Evaluación comparativa de configuraciones
- `--all`: Todas las evaluaciones
- `--no-reports`: No guardar reportes
- `--verbose`: Salida detallada

## 🛠️ Scripts de Utilidades

### `dev_utils.sh`
**Utilidades de desarrollo y testing**
```bash
./dev_utils.sh <comando>
```

**Comandos disponibles:**
- `test-api`: Probar endpoints de la API
- `debug`: Mostrar estado completo del sistema
- `quick-test`: Test rápido de funcionalidad
- `benchmark`: Benchmark de rendimiento
- `logs`: Mostrar logs del sistema
- `reindex`: Re-indexar documentos

**Ejemplos:**
```bash
./dev_utils.sh debug        # Ver estado del sistema
./dev_utils.sh test-api     # Probar API (requiere servidor activo)
./dev_utils.sh benchmark    # Medir rendimiento
```

### `cleanup.sh`
**Limpieza y mantenimiento del sistema**
```bash
./cleanup.sh [opciones]
```
- Limpia archivos temporales y cache
- Gestiona logs antiguos
- Elimina reportes obsoletos
- Optimiza y mantiene el sistema
- Modo dry-run para vista previa

**Opciones:**
- `--cache`: Limpiar cache de Python
- `--logs`: Limpiar logs antiguos
- `--reports`: Limpiar reportes antiguos
- `--index`: ⚠️ Eliminar índice vectorial
- `--venv`: ⚠️ Eliminar entorno virtual
- `--all`: Limpieza completa segura
- `--dry-run`: Mostrar qué se eliminaría
- `--no-confirm`: No pedir confirmación

## 📋 Flujos de Trabajo Recomendados

### Para Nuevos Usuarios
```bash
# Inicio rápido todo-en-uno
./quick_start.sh
```

### Para Desarrollo
```bash
# Configuración inicial
./setup.sh

# Desarrollo local diario
./start_local.sh

# Testing y debugging
./dev_utils.sh debug
./dev_utils.sh quick-test
```

### Para Producción
```bash
# Configurar entorno
./setup.sh

# Evaluar sistema
./run_evaluation.sh --all

# Desplegar en producción
./start_production.sh --port 8080 --workers 4
```

### Para Mantenimiento
```bash
# Evaluar rendimiento
./run_evaluation.sh --performance

# Limpiar sistema
./cleanup.sh --all

# Re-indexar si es necesario
./dev_utils.sh reindex
```

## 🔧 Configuración

### Variables de Entorno Importantes
Ver archivo `.env.example` para configuración completa.

**Básicas:**
- `OPENAI_API_KEY`: Clave de API de OpenAI
- `LLM_PROVIDER`: openai | ollama | huggingface
- `EMBEDDING_PROVIDER`: openai | ollama | huggingface

**Avanzadas:**
- `CHUNK_SIZE`: Tamaño de chunks (default: 512)
- `CHUNK_OVERLAP`: Overlap de chunks (default: 51)
- `PORT`: Puerto del servidor (default: 5000)

### Archivos de Configuración
- `.env`: Configuración principal
- `.env.local`: Generado automáticamente para desarrollo local
- `.env.production`: Generado para producción
- `requirements.txt`: Dependencias de Python

## 📚 Ayuda y Documentación

Todos los scripts incluyen ayuda detallada:
```bash
./script_name.sh --help
```

**Documentos adicionales:**
- `README.md`: Documentación principal del proyecto
- `TECHNICAL_ANALYSIS.md`: Análisis técnico detallado
- `requirements.txt`: Dependencias con comentarios

## 🚨 Notas Importantes

1. **Siempre ejecuta `./setup.sh` antes del primer uso**
2. **Para desarrollo local**: Instala Ollama desde https://ollama.ai/
3. **Para producción**: Configura `OPENAI_API_KEY` en `.env`
4. **Los scripts con ⚠️ requieren confirmación** (usa `--no-confirm` para automatizar)
5. **Usa `--dry-run` para preview** antes de operaciones destructivas

## 🆘 Solución de Problemas

**Error: "Entorno virtual no encontrado"**
```bash
./setup.sh
```

**Error: "Ollama no está instalado"**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Error: "OpenAI API key no configurada"**
```bash
# Edita .env y agrega tu clave
OPENAI_API_KEY=sk-tu-clave-aqui
```

**Sistema lento o con errores**
```bash
./dev_utils.sh debug
./cleanup.sh --all
``` 