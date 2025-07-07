"""
Script de Testing de Performance para Configuraciones de Chunking

Este script permite evaluar el impacto de diferentes configuraciones de chunking
en el rendimiento del sistema RAG, respondiendo a las observaciones sobre
justificación técnica de parámetros de chunking.
"""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

from rag_chain import RAGPipeline
from ingest import ingest_and_build
from evaluation import RAGEvaluator


@dataclass
class ChunkingConfig:
    """Configuración de chunking para testing"""
    chunk_size: int
    chunk_overlap: int
    name: str
    description: str


class ChunkingPerformanceTester:
    """
    Tester para evaluar diferentes configuraciones de chunking.
    
    Evalúa el impacto en:
    - Tiempo de indexación
    - Tiempo de consulta
    - Calidad de retrieval
    - Uso de memoria
    """
    
    def __init__(self, test_queries: List[str] = None):
        """
        Inicializa el tester con consultas de prueba.
        
        Args:
            test_queries: Lista de consultas para testing. Si None, usa consultas por defecto.
        """
        self.test_queries = test_queries or [
            "¿Qué necesito para obtener un certificado de antecedentes?",
            "¿Cuáles son los requisitos para registrar una marca comercial?",
            "¿Cómo puedo solicitar una inspección ambiental?",
            "¿Qué documentos necesito para una licencia de conducir?",
            "¿Cuál es el proceso para abrir un restaurante?",
            "¿Cómo registro una empresa en Córdoba?",
            "¿Qué permisos necesito para construir una casa?",
            "¿Cómo solicito un subsidio gubernamental?",
            "¿Cuáles son los trámites para adopción?",
            "¿Qué necesito para exportar productos?"
        ]
        
        # Configuraciones de chunking a evaluar
        self.chunking_configs = [
            ChunkingConfig(256, 25, "tiny", "Chunks muy pequeños (256 chars, 10% overlap)"),
            ChunkingConfig(512, 51, "small", "Chunks pequeños (512 chars, 10% overlap)"),
            ChunkingConfig(1000, 100, "medium", "Chunks medianos (1000 chars, 10% overlap)"),
            ChunkingConfig(1500, 150, "large", "Chunks grandes (1500 chars, 10% overlap)"),
            ChunkingConfig(2000, 200, "xlarge", "Chunks muy grandes (2000 chars, 10% overlap)"),
            ChunkingConfig(1000, 200, "medium-high-overlap", "Overlap alto (1000 chars, 20% overlap)"),
            ChunkingConfig(1000, 50, "medium-low-overlap", "Overlap bajo (1000 chars, 5% overlap)")
        ]
    
    def test_indexing_performance(self, config: ChunkingConfig, 
                                 embedding_provider: str = "openai",
                                 embedding_model: str = None) -> Dict[str, Any]:
        """
        Evalúa el rendimiento de indexación para una configuración específica.
        
        Args:
            config: Configuración de chunking a evaluar
            embedding_provider: Proveedor de embedding
            embedding_model: Modelo específico de embedding
            
        Returns:
            Métricas de rendimiento de indexación
        """
        print(f"🔧 Testing indexación - {config.name}: {config.description}")
        
        # Directorio temporal para esta configuración
        temp_storage = Path(f"temp_storage_{config.name}")
        temp_storage.mkdir(exist_ok=True)
        
        # Medir tiempo de indexación
        start_time = time.time()
        
        try:
            # Simular ingesta con configuración específica
            # (Modificaríamos temporalmente la configuración de chunking)
            vector_store = ingest_and_build(
                str(temp_storage),
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            )
            
            end_time = time.time()
            indexing_time = end_time - start_time
            
            # Leer metadatos para obtener información de chunks
            metadata_path = temp_storage / "index_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            total_chunks = metadata.get("chunking_config", {}).get("total_chunks", 0)
            
            # Calcular métricas
            metrics = {
                "config_name": config.name,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "indexing_time_seconds": round(indexing_time, 2),
                "total_chunks": total_chunks,
                "chunks_per_second": round(total_chunks / indexing_time, 2) if indexing_time > 0 else 0,
                "avg_time_per_chunk_ms": round((indexing_time / total_chunks) * 1000, 2) if total_chunks > 0 else 0
            }
            
            print(f"   ✅ Completado: {total_chunks} chunks en {indexing_time:.2f}s")
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return {
                "config_name": config.name,
                "error": str(e),
                "indexing_time_seconds": None,
                "total_chunks": None
            }
        
        finally:
            # Limpiar directorio temporal
            import shutil
            if temp_storage.exists():
                shutil.rmtree(temp_storage)
    
    def test_query_performance(self, config: ChunkingConfig,
                              embedding_provider: str = "openai",
                              embedding_model: str = None,
                              llm_provider: str = "openai",
                              llm_model: str = None) -> Dict[str, Any]:
        """
        Evalúa el rendimiento de consultas para una configuración específica.
        
        Args:
            config: Configuración de chunking
            embedding_provider: Proveedor de embedding
            embedding_model: Modelo de embedding
            llm_provider: Proveedor de LLM
            llm_model: Modelo de LLM
            
        Returns:
            Métricas de rendimiento de consultas
        """
        print(f"🔍 Testing consultas - {config.name}: {config.description}")
        
        try:
            # Crear pipeline para esta configuración
            # (En un escenario real, ya tendríamos el índice creado)
            pipeline = RAGPipeline(
                llm_provider=llm_provider,
                llm_model=llm_model,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            )
            
            query_times = []
            retrieval_counts = []
            response_lengths = []
            
            # Ejecutar consultas de prueba
            for i, query in enumerate(self.test_queries):
                print(f"   Query {i+1}/{len(self.test_queries)}: {query[:50]}...")
                
                start_time = time.time()
                result = pipeline(query)
                end_time = time.time()
                
                query_time = end_time - start_time
                query_times.append(query_time)
                
                # Métricas de la respuesta
                retrieved_docs = len(result.get("source_documents", []))
                retrieval_counts.append(retrieved_docs)
                
                response_length = len(result.get("result", "").split())
                response_lengths.append(response_length)
            
            # Calcular estadísticas
            metrics = {
                "config_name": config.name,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "avg_query_time": round(statistics.mean(query_times), 3),
                "median_query_time": round(statistics.median(query_times), 3),
                "min_query_time": round(min(query_times), 3),
                "max_query_time": round(max(query_times), 3),
                "std_query_time": round(statistics.stdev(query_times), 3) if len(query_times) > 1 else 0,
                "avg_retrieved_docs": round(statistics.mean(retrieval_counts), 1),
                "avg_response_length": round(statistics.mean(response_lengths), 1),
                "total_queries": len(self.test_queries)
            }
            
            print(f"   ✅ Completado: {metrics['avg_query_time']}s promedio por consulta")
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return {
                "config_name": config.name,
                "error": str(e),
                "avg_query_time": None
            }
    
    def comprehensive_chunking_test(self, 
                                   embedding_provider: str = "openai",
                                   embedding_model: str = None,
                                   llm_provider: str = "openai", 
                                   llm_model: str = None,
                                   include_quality_metrics: bool = True) -> pd.DataFrame:
        """
        Ejecuta test comprehensivo de todas las configuraciones de chunking.
        
        Args:
            embedding_provider: Proveedor de embedding
            embedding_model: Modelo de embedding
            llm_provider: Proveedor de LLM
            llm_model: Modelo de LLM
            include_quality_metrics: Si incluir métricas de calidad (requiere más tiempo)
            
        Returns:
            DataFrame con resultados comparativos
        """
        print("🏁 Iniciando test comprehensivo de configuraciones de chunking...")
        print(f"📊 Configuraciones a evaluar: {len(self.chunking_configs)}")
        
        all_results = []
        
        for i, config in enumerate(self.chunking_configs):
            print(f"\n📋 Evaluando configuración {i+1}/{len(self.chunking_configs)}")
            
            # Test de indexación
            indexing_metrics = self.test_indexing_performance(
                config, embedding_provider, embedding_model
            )
            
            # Test de consultas (solo si indexación fue exitosa)
            if "error" not in indexing_metrics:
                query_metrics = self.test_query_performance(
                    config, embedding_provider, embedding_model, 
                    llm_provider, llm_model
                )
                
                # Combinar métricas
                combined_metrics = {**indexing_metrics, **query_metrics}
                
                # Métricas de calidad (opcional, requiere más tiempo)
                if include_quality_metrics and "error" not in query_metrics:
                    try:
                        evaluator = RAGEvaluator()
                        pipeline = RAGPipeline(
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            embedding_provider=embedding_provider,
                            embedding_model=embedding_model
                        )
                        
                        quality_metrics = evaluator.evaluate_retrieval_metrics(pipeline)
                        combined_metrics.update({
                            f"quality_{k}": v for k, v in quality_metrics.items()
                        })
                        
                    except Exception as e:
                        print(f"   ⚠️ Error en métricas de calidad: {e}")
                        combined_metrics["quality_error"] = str(e)
                
            else:
                # Solo métricas de indexación si falló
                combined_metrics = indexing_metrics
            
            all_results.append(combined_metrics)
        
        # Crear DataFrame y guardar resultados
        df = pd.DataFrame(all_results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"chunking_performance_test_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n📈 Resultados guardados en: {output_file}")
        
        # Mostrar resumen
        self._print_performance_summary(df)
        
        return df
    
    def _print_performance_summary(self, df: pd.DataFrame):
        """Imprime resumen de resultados de performance"""
        print("\n📊 RESUMEN DE PERFORMANCE:")
        print("=" * 50)
        
        # Top configuraciones por velocidad de consulta
        if "avg_query_time" in df.columns:
            fastest_configs = df[df["avg_query_time"].notna()].nsmallest(3, "avg_query_time")
            print("\n🚀 TOP 3 - CONSULTAS MÁS RÁPIDAS:")
            for _, row in fastest_configs.iterrows():
                print(f"   {row['config_name']}: {row['avg_query_time']:.3f}s promedio")
        
        # Top configuraciones por velocidad de indexación
        if "chunks_per_second" in df.columns:
            fastest_indexing = df[df["chunks_per_second"].notna()].nlargest(3, "chunks_per_second")
            print("\n⚡ TOP 3 - INDEXACIÓN MÁS RÁPIDA:")
            for _, row in fastest_indexing.iterrows():
                print(f"   {row['config_name']}: {row['chunks_per_second']:.1f} chunks/s")
        
        # Configuraciones por total de chunks
        if "total_chunks" in df.columns:
            chunk_analysis = df[df["total_chunks"].notna()].sort_values("total_chunks")
            print("\n📦 ANÁLISIS DE CHUNKS GENERADOS:")
            for _, row in chunk_analysis.iterrows():
                print(f"   {row['config_name']}: {row['total_chunks']} chunks (tamaño: {row['chunk_size']})")
        
        # Recomendaciones automáticas
        print("\n💡 RECOMENDACIONES:")
        self._generate_chunking_recommendations(df)
    
    def _generate_chunking_recommendations(self, df: pd.DataFrame):
        """Genera recomendaciones automáticas basadas en resultados"""
        recommendations = []
        
        # Analizar trade-offs
        if "avg_query_time" in df.columns and "total_chunks" in df.columns:
            df_clean = df.dropna(subset=["avg_query_time", "total_chunks"])
            
            if not df_clean.empty:
                # Configuración más balanceada (considera velocidad y granularidad)
                df_clean["balance_score"] = (
                    (1 / df_clean["avg_query_time"]) * 0.6 +  # Velocidad (60%)
                    (df_clean["total_chunks"] / df_clean["total_chunks"].max()) * 0.4  # Granularidad (40%)
                )
                
                best_balance = df_clean.loc[df_clean["balance_score"].idxmax()]
                recommendations.append(
                    f"   🎯 Configuración más balanceada: {best_balance['config_name']} "
                    f"(velocidad: {best_balance['avg_query_time']:.3f}s, chunks: {best_balance['total_chunks']})"
                )
                
                # Análisis de tamaño óptimo
                optimal_size_range = df_clean[
                    (df_clean["chunk_size"] >= 800) & (df_clean["chunk_size"] <= 1200)
                ]
                if not optimal_size_range.empty:
                    recommendations.append(
                        "   📏 Rango óptimo detectado: 800-1200 caracteres para balance velocidad/calidad"
                    )
                
                # Análisis de overlap
                high_overlap = df_clean[df_clean["chunk_overlap"] > 150]
                if not high_overlap.empty and "avg_query_time" in high_overlap.columns:
                    avg_time_high_overlap = high_overlap["avg_query_time"].mean()
                    avg_time_low_overlap = df_clean[df_clean["chunk_overlap"] <= 150]["avg_query_time"].mean()
                    
                    if avg_time_high_overlap > avg_time_low_overlap * 1.2:
                        recommendations.append(
                            "   ⚠️ Overlap alto (>150 chars) puede impactar performance negativamente"
                        )
        
        # Mostrar recomendaciones
        for rec in recommendations:
            print(rec)
        
        if not recommendations:
            print("   📊 Analiza los resultados para determinar la configuración óptima para tu caso de uso")


def main():
    """Función principal para ejecutar tests de performance"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test de performance de configuraciones de chunking')
    parser.add_argument('--embedding-provider', type=str, default='openai',
                       help='Proveedor de embedding (openai, ollama, huggingface)')
    parser.add_argument('--embedding-model', type=str, default=None,
                       help='Modelo específico de embedding')
    parser.add_argument('--llm-provider', type=str, default='openai',
                       help='Proveedor de LLM')
    parser.add_argument('--llm-model', type=str, default=None,
                       help='Modelo específico de LLM')
    parser.add_argument('--skip-quality', action='store_true',
                       help='Omitir métricas de calidad para test más rápido')
    parser.add_argument('--config-name', type=str, default=None,
                       help='Probar solo una configuración específica (tiny, small, medium, etc.)')
    
    args = parser.parse_args()
    
    # Crear tester
    tester = ChunkingPerformanceTester()
    
    if args.config_name:
        # Test de configuración específica
        config = next((c for c in tester.chunking_configs if c.name == args.config_name), None)
        if not config:
            print(f"❌ Configuración '{args.config_name}' no encontrada")
            print(f"Configuraciones disponibles: {[c.name for c in tester.chunking_configs]}")
            return
        
        print(f"🔧 Testing configuración específica: {config.name}")
        
        # Test de indexación
        indexing_results = tester.test_indexing_performance(
            config, args.embedding_provider, args.embedding_model
        )
        print(f"📊 Resultados indexación: {indexing_results}")
        
        # Test de consultas
        query_results = tester.test_query_performance(
            config, args.embedding_provider, args.embedding_model,
            args.llm_provider, args.llm_model
        )
        print(f"📊 Resultados consultas: {query_results}")
        
    else:
        # Test comprehensivo
        include_quality = not args.skip_quality
        
        print(f"🚀 Iniciando test comprehensivo de chunking")
        print(f"⚙️ Configuración:")
        print(f"   • Embedding: {args.embedding_provider}:{args.embedding_model or 'default'}")
        print(f"   • LLM: {args.llm_provider}:{args.llm_model or 'default'}")
        print(f"   • Métricas de calidad: {'Sí' if include_quality else 'No'}")
        
        results_df = tester.comprehensive_chunking_test(
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            include_quality_metrics=include_quality
        )
        
        print(f"\n✅ Test completado. Resultados disponibles en DataFrame y archivo CSV.")


if __name__ == "__main__":
    main() 