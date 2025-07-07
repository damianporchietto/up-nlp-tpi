"""
Framework de Evaluación Robusto para RAG

Este módulo implementa métricas cuantitativas y evaluación sistemática
para responder a las observaciones sobre robustecimiento de la evaluación pre-producción.

Incluye:
- Métricas de retrieval (Recall@K, Precision@K, MRR)
- Métricas de generación (BLEU, ROUGE, similitud semántica)
- Evaluación de consistencia de chunking
- Dataset de evaluación expandible
- Benchmarking de diferentes configuraciones
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from rag_chain import RAGPipeline, load_rag_chain
from model_providers import get_embeddings_model


@dataclass
class EvaluationQuery:
    """Estructura para consultas de evaluación"""
    query: str
    expected_answer: str
    relevant_doc_ids: List[str]  # IDs de documentos relevantes conocidos
    category: str  # Categoría del trámite
    difficulty: str  # easy, medium, hard


class RAGEvaluator:
    """
    Evaluador comprehensivo para sistemas RAG.
    
    Resuelve las observaciones sobre evaluación:
    - Métricas cuantitativas robustas
    - Dataset de prueba ampliado
    - Evaluación sistemática de diferentes configuraciones
    """
    
    def __init__(self, evaluation_data_path: Optional[str] = None):
        """
        Inicializa el evaluador con dataset de evaluación.
        
        Args:
            evaluation_data_path: Ruta al archivo JSON con queries de evaluación
        """
        self.evaluation_data_path = evaluation_data_path or "evaluation_dataset.json"
        self.evaluation_queries = self._load_evaluation_dataset()
        
    def _load_evaluation_dataset(self) -> List[EvaluationQuery]:
        """Carga el dataset de evaluación expandido"""
        try:
            with open(self.evaluation_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return [
                EvaluationQuery(
                    query=item["query"],
                    expected_answer=item["expected_answer"],
                    relevant_doc_ids=item.get("relevant_doc_ids", []),
                    category=item.get("category", "general"),
                    difficulty=item.get("difficulty", "medium")
                )
                for item in data
            ]
        except FileNotFoundError:
            print(f"⚠️ Dataset de evaluación no encontrado en {self.evaluation_data_path}")
            print("🔧 Creando dataset básico de ejemplo...")
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> List[EvaluationQuery]:
        """Crea un dataset de evaluación básico para empezar"""
        sample_queries = [
            EvaluationQuery(
                query="¿Qué necesito para obtener un certificado de antecedentes?",
                expected_answer="Necesitas DNI en perfecto estado, comprobante de pago de la Tasa Retributiva de Servicio, tener domicilio en Córdoba y realizar el trámite de forma personal.",
                relevant_doc_ids=["certificado_antecedentes"],
                category="certificados",
                difficulty="easy"
            ),
            EvaluationQuery(
                query="¿Cuáles son los requisitos para registrar una marca comercial?",
                expected_answer="Debes presentar solicitud completa, comprobante de pago de tasas, documentación de identidad, y descripción detallada de la marca.",
                relevant_doc_ids=["registro_marca"],
                category="comercial",
                difficulty="medium"
            ),
            EvaluationQuery(
                query="¿Cómo puedo solicitar una inspección ambiental para mi empresa?",
                expected_answer="Debes completar formulario de solicitud, presentar documentación técnica del proyecto, pagar las tasas correspondientes y cumplir con normativas ambientales.",
                relevant_doc_ids=["inspeccion_ambiental"],
                category="ambiente",
                difficulty="hard"
            )
        ]
        
        # Guardar dataset de ejemplo
        sample_data = [
            {
                "query": q.query,
                "expected_answer": q.expected_answer,
                "relevant_doc_ids": q.relevant_doc_ids,
                "category": q.category,
                "difficulty": q.difficulty
            }
            for q in sample_queries
        ]
        
        with open(self.evaluation_data_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset de ejemplo creado en {self.evaluation_data_path}")
        return sample_queries
    
    def evaluate_retrieval_metrics(self, pipeline: RAGPipeline, k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        Evalúa métricas de retrieval: Recall@K, Precision@K, MRR
        
        Args:
            pipeline: Pipeline RAG a evaluar
            k_values: Valores de K para evaluar Recall@K y Precision@K
            
        Returns:
            Diccionario con métricas de retrieval
        """
        recall_scores = {f"recall@{k}": [] for k in k_values}
        precision_scores = {f"precision@{k}": [] for k in k_values}
        mrr_scores = []
        
        for query_obj in self.evaluation_queries:
            if not query_obj.relevant_doc_ids:
                continue
                
            # Obtener documentos recuperados
            retrieved_docs = pipeline.retriever.get_relevant_documents(query_obj.query)
            retrieved_doc_ids = [
                doc.metadata.get("source", "").split("/")[-1].replace(".json", "")
                for doc in retrieved_docs
            ]
            
            # Calcular métricas para cada K
            for k in k_values:
                retrieved_k = retrieved_doc_ids[:k]
                relevant_retrieved = set(retrieved_k) & set(query_obj.relevant_doc_ids)
                
                # Recall@K: fracción de documentos relevantes recuperados
                recall = len(relevant_retrieved) / len(query_obj.relevant_doc_ids)
                recall_scores[f"recall@{k}"].append(recall)
                
                # Precision@K: fracción de documentos recuperados que son relevantes
                precision = len(relevant_retrieved) / k if k > 0 else 0
                precision_scores[f"precision@{k}"].append(precision)
            
            # MRR: Mean Reciprocal Rank
            rank = None
            for i, doc_id in enumerate(retrieved_doc_ids):
                if doc_id in query_obj.relevant_doc_ids:
                    rank = i + 1
                    break
            
            mrr_scores.append(1.0 / rank if rank else 0.0)
        
        # Promediar métricas
        metrics = {}
        for metric, scores in {**recall_scores, **precision_scores}.items():
            metrics[metric] = statistics.mean(scores) if scores else 0.0
        
        metrics["mrr"] = statistics.mean(mrr_scores) if mrr_scores else 0.0
        
        return metrics
    
    def evaluate_generation_quality(self, pipeline: RAGPipeline) -> Dict[str, float]:
        """
        Evalúa calidad de generación usando similitud semántica
        (BLEU y ROUGE requieren librerías adicionales)
        
        Args:
            pipeline: Pipeline RAG a evaluar
            
        Returns:
            Diccionario con métricas de generación
        """
        similarity_scores = []
        response_lengths = []
        response_times = []
        
        # Obtener modelo de embeddings para calcular similitud
        embeddings_model = get_embeddings_model(
            provider=pipeline.index_metadata.get("embedding_provider", "openai"),
            model_name=pipeline.index_metadata.get("embedding_model")
        )
        
        for query_obj in self.evaluation_queries:
            start_time = time.time()
            
            # Generar respuesta
            result = pipeline(query_obj.query)
            generated_answer = result["result"]
            
            end_time = time.time()
            response_times.append(end_time - start_time)
            response_lengths.append(len(generated_answer.split()))
            
            # Calcular similitud semántica con respuesta esperada
            try:
                expected_embedding = embeddings_model.embed_query(query_obj.expected_answer)
                generated_embedding = embeddings_model.embed_query(generated_answer)
                
                similarity = cosine_similarity(
                    [expected_embedding], 
                    [generated_embedding]
                )[0][0]
                similarity_scores.append(similarity)
                
            except Exception as e:
                print(f"⚠️ Error calculando similitud para query '{query_obj.query[:50]}...': {e}")
                similarity_scores.append(0.0)
        
        return {
            "semantic_similarity_mean": statistics.mean(similarity_scores) if similarity_scores else 0.0,
            "semantic_similarity_std": statistics.stdev(similarity_scores) if len(similarity_scores) > 1 else 0.0,
            "avg_response_length": statistics.mean(response_lengths) if response_lengths else 0.0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0.0,
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0.0
        }
    
    def evaluate_chunking_consistency(self, pipeline: RAGPipeline) -> Dict[str, Any]:
        """
        Evalúa la consistencia y efectividad de la estrategia de chunking
        
        Args:
            pipeline: Pipeline RAG a evaluar
            
        Returns:
            Métricas de chunking
        """
        chunk_metrics = pipeline.index_metadata.get("chunking_config", {})
        
        # Analizar distribución de chunks recuperados
        chunks_retrieved_per_query = []
        unique_sources_per_query = []
        
        for query_obj in self.evaluation_queries:
            retrieved_docs = pipeline.retriever.get_relevant_documents(query_obj.query)
            chunks_retrieved_per_query.append(len(retrieved_docs))
            
            unique_sources = set(doc.metadata.get("source", "") for doc in retrieved_docs)
            unique_sources_per_query.append(len(unique_sources))
        
        return {
            "chunk_size_chars": chunk_metrics.get("chunk_size_chars", 0),
            "chunk_overlap_chars": chunk_metrics.get("chunk_overlap_chars", 0),
            "overlap_percentage": chunk_metrics.get("overlap_percentage", 0),
            "total_chunks": chunk_metrics.get("total_chunks", 0),
            "avg_chunks_retrieved": statistics.mean(chunks_retrieved_per_query) if chunks_retrieved_per_query else 0,
            "avg_unique_sources": statistics.mean(unique_sources_per_query) if unique_sources_per_query else 0,
            "retrieval_diversity": statistics.mean(unique_sources_per_query) / statistics.mean(chunks_retrieved_per_query) if chunks_retrieved_per_query and all(x > 0 for x in chunks_retrieved_per_query) else 0
        }
    
    def comprehensive_evaluation(self, pipeline: RAGPipeline) -> Dict[str, Any]:
        """
        Ejecuta evaluación comprehensiva del pipeline RAG
        
        Args:
            pipeline: Pipeline RAG a evaluar
            
        Returns:
            Resultados completos de evaluación
        """
        print("🔍 Iniciando evaluación comprehensiva del RAG...")
        
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_config": {
                "embedding_model": pipeline.index_metadata.get("embedding_model"),
                "embedding_provider": pipeline.index_metadata.get("embedding_provider"),
                "preprocessing_strategy": pipeline.index_metadata.get("preprocessing_strategy"),
                "chunking_config": pipeline.index_metadata.get("chunking_config")
            },
            "dataset_info": {
                "total_queries": len(self.evaluation_queries),
                "categories": list(set(q.category for q in self.evaluation_queries)),
                "difficulty_distribution": {
                    difficulty: len([q for q in self.evaluation_queries if q.difficulty == difficulty])
                    for difficulty in ["easy", "medium", "hard"]
                }
            }
        }
        
        # Evaluar métricas de retrieval
        print("📊 Evaluando métricas de retrieval...")
        retrieval_metrics = self.evaluate_retrieval_metrics(pipeline)
        evaluation_results["retrieval_metrics"] = retrieval_metrics
        
        # Evaluar calidad de generación
        print("✍️ Evaluando calidad de generación...")
        generation_metrics = self.evaluate_generation_quality(pipeline)
        evaluation_results["generation_metrics"] = generation_metrics
        
        # Evaluar chunking
        print("🔧 Evaluando estrategia de chunking...")
        chunking_metrics = self.evaluate_chunking_consistency(pipeline)
        evaluation_results["chunking_metrics"] = chunking_metrics
        
        # Calcular score general
        overall_score = self._calculate_overall_score(retrieval_metrics, generation_metrics)
        evaluation_results["overall_score"] = overall_score
        
        print("✅ Evaluación comprehensiva completada")
        return evaluation_results
    
    def _calculate_overall_score(self, retrieval_metrics: Dict, generation_metrics: Dict) -> float:
        """Calcula un score general ponderado"""
        # Ponderación: 40% retrieval, 60% generación
        retrieval_score = (
            retrieval_metrics.get("recall@3", 0) * 0.4 +
            retrieval_metrics.get("precision@3", 0) * 0.3 +
            retrieval_metrics.get("mrr", 0) * 0.3
        )
        
        generation_score = generation_metrics.get("semantic_similarity_mean", 0)
        
        overall_score = retrieval_score * 0.4 + generation_score * 0.6
        return round(overall_score, 3)
    
    def benchmark_configurations(self, configurations: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Compara múltiples configuraciones de RAG
        
        Args:
            configurations: Lista de configuraciones a comparar
                           [{"llm_provider": "openai", "embedding_provider": "openai", ...}, ...]
            
        Returns:
            DataFrame con resultados comparativos
        """
        print("🏁 Iniciando benchmark de configuraciones...")
        
        results = []
        
        for i, config in enumerate(configurations):
            print(f"\n📋 Evaluando configuración {i+1}/{len(configurations)}: {config}")
            
            try:
                # Crear pipeline con configuración específica
                pipeline = RAGPipeline(
                    llm_provider=config.get("llm_provider", "openai"),
                    llm_model=config.get("llm_model"),
                    embedding_provider=config.get("embedding_provider", "openai"),
                    embedding_model=config.get("embedding_model")
                )
                
                # Evaluar
                eval_results = self.comprehensive_evaluation(pipeline)
                
                # Agregar configuración a resultados
                result_row = {
                    "config_id": f"config_{i+1}",
                    "llm_provider": config.get("llm_provider"),
                    "llm_model": config.get("llm_model", "default"),
                    "embedding_provider": config.get("embedding_provider"),
                    "embedding_model": config.get("embedding_model", "default"),
                    "overall_score": eval_results["overall_score"],
                    **eval_results["retrieval_metrics"],
                    **eval_results["generation_metrics"],
                    "total_chunks": eval_results["chunking_metrics"]["total_chunks"],
                    "avg_response_time": eval_results["generation_metrics"]["avg_response_time"]
                }
                
                results.append(result_row)
                
            except Exception as e:
                print(f"❌ Error evaluando configuración {config}: {e}")
                # Agregar resultado con error
                results.append({
                    "config_id": f"config_{i+1}",
                    "error": str(e),
                    **config
                })
        
        df = pd.DataFrame(results)
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"📈 Resultados del benchmark guardados en: {output_file}")
        return df
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], output_path: str = None) -> str:
        """
        Genera un reporte detallado de evaluación
        
        Args:
            evaluation_results: Resultados de evaluación comprehensiva
            output_path: Ruta donde guardar el reporte
            
        Returns:
            Ruta del archivo de reporte generado
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_report_{timestamp}.json"
        
        # Agregar análisis y recomendaciones
        evaluation_results["analysis"] = self._generate_analysis(evaluation_results)
        evaluation_results["recommendations"] = self._generate_recommendations(evaluation_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Reporte de evaluación guardado en: {output_path}")
        return output_path
    
    def _generate_analysis(self, results: Dict) -> Dict[str, str]:
        """Genera análisis automático de resultados"""
        retrieval = results.get("retrieval_metrics", {})
        generation = results.get("generation_metrics", {})
        chunking = results.get("chunking_metrics", {})
        
        analysis = {}
        
        # Análisis de retrieval
        recall_3 = retrieval.get("recall@3", 0)
        if recall_3 > 0.8:
            analysis["retrieval"] = "Excelente: El sistema recupera la mayoría de documentos relevantes"
        elif recall_3 > 0.6:
            analysis["retrieval"] = "Bueno: Retrieval efectivo pero con margen de mejora"
        else:
            analysis["retrieval"] = "Necesita mejora: Considerar ajustar chunking o modelo de embedding"
        
        # Análisis de generación
        similarity = generation.get("semantic_similarity_mean", 0)
        if similarity > 0.8:
            analysis["generation"] = "Excelente: Respuestas altamente coherentes con referencias"
        elif similarity > 0.6:
            analysis["generation"] = "Bueno: Calidad de generación aceptable"
        else:
            analysis["generation"] = "Necesita mejora: Considerar modelo LLM más potente o prompt engineering"
        
        # Análisis de performance
        avg_time = generation.get("avg_response_time", 0)
        if avg_time < 2.0:
            analysis["performance"] = "Excelente: Tiempos de respuesta rápidos"
        elif avg_time < 5.0:
            analysis["performance"] = "Aceptable: Tiempos de respuesta moderados"
        else:
            analysis["performance"] = "Lento: Considerar optimización o modelos más eficientes"
        
        return analysis
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Genera recomendaciones automáticas"""
        recommendations = []
        
        retrieval = results.get("retrieval_metrics", {})
        generation = results.get("generation_metrics", {})
        chunking = results.get("chunking_metrics", {})
        
        # Recomendaciones basadas en métricas
        if retrieval.get("recall@3", 0) < 0.6:
            recommendations.append("Mejorar retrieval: Considerar chunking más granular o modelo de embedding más potente")
        
        if generation.get("semantic_similarity_mean", 0) < 0.6:
            recommendations.append("Mejorar generación: Evaluar modelo LLM más avanzado o optimizar prompt")
        
        if generation.get("avg_response_time", 0) > 5.0:
            recommendations.append("Optimizar performance: Considerar modelos más eficientes o cache de respuestas")
        
        if chunking.get("retrieval_diversity", 0) < 0.5:
            recommendations.append("Diversificar retrieval: Ajustar parámetros de búsqueda para mayor variedad de fuentes")
        
        if not recommendations:
            recommendations.append("Sistema funcionando bien: Continuar monitoreo y considerar expansión de dataset de evaluación")
        
        return recommendations


def main():
    """Función principal para ejecutar evaluaciones"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluación comprehensiva de RAG')
    parser.add_argument('--config-file', type=str, 
                       help='Archivo JSON con configuraciones a benchmarcar')
    parser.add_argument('--evaluation-data', type=str, default='evaluation_dataset.json',
                       help='Archivo con dataset de evaluación')
    parser.add_argument('--single-eval', action='store_true',
                       help='Evaluar configuración actual únicamente')
    
    args = parser.parse_args()
    
    # Crear evaluador
    evaluator = RAGEvaluator(args.evaluation_data)
    
    if args.single_eval:
        # Evaluación simple de configuración actual
        pipeline = load_rag_chain()
        results = evaluator.comprehensive_evaluation(pipeline)
        report_path = evaluator.generate_evaluation_report(results)
        
        print(f"\n📊 RESUMEN DE EVALUACIÓN:")
        print(f"Score General: {results['overall_score']:.3f}")
        print(f"Recall@3: {results['retrieval_metrics']['recall@3']:.3f}")
        print(f"Similitud Semántica: {results['generation_metrics']['semantic_similarity_mean']:.3f}")
        print(f"Tiempo Promedio: {results['generation_metrics']['avg_response_time']:.2f}s")
        
    elif args.config_file:
        # Benchmark de múltiples configuraciones
        with open(args.config_file, 'r') as f:
            configurations = json.load(f)
        
        df_results = evaluator.benchmark_configurations(configurations)
        print(f"\n📈 Top 3 configuraciones:")
        print(df_results.nlargest(3, 'overall_score')[['config_id', 'overall_score', 'recall@3', 'semantic_similarity_mean']])
    
    else:
        print("Especifica --single-eval o --config-file para ejecutar evaluación")


if __name__ == "__main__":
    main() 