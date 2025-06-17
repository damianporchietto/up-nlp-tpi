import json
import time
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Questions to test
QUESTIONS = [
    "¿Qué trámite permite afiliar a un abuelo dentro del grupo familiar?",
    "¿Qué documentación se necesita para afiliar a un ex cónyuge?",
    "¿Cuál es el nivel de cuenta requerido en Ciudadano Digital para realizar la afiliación de cónyuge sin obra social?",
    "¿Se requiere renovación para la afiliación de abuelos a APROSS?",
    "¿Qué pasa si no se renueva la afiliación del cónyuge sin obra social?",
    "¿Qué se necesita para acceder a la credencial digital de un familiar?",
    "¿Qué diferencia hay entre la afiliación de un cónyuge y de un ex cónyuge respecto a los aportes?",
    "¿Qué trámite se debe hacer para dar de baja a un familiar a cargo?",
    "¿Qué documentación se exige para dar de baja a un familiar fallecido?",
    "¿Es obligatorio que el representante que accede a la credencial digital esté afiliado a APROSS?",
    "¿Qué tipo de cuenta digital se necesita para cualquier trámite relacionado con afiliaciones?",
    "¿Quién puede solicitar la afiliación de un ex cónyuge?",
    "¿Qué significa que los aportes afiliatorios son \"a mes completo\"?",
    "¿Qué documentación respalda la afiliación de un cónyuge?",
    "¿Qué trámite se utiliza para habilitar a alguien a ver la credencial digital de otro familiar?",
    "¿Hay restricción de prestaciones para la afiliación de ex cónyuges?",
    "¿Se puede usar el mismo trámite para afiliar a cualquier familiar?",
    "¿Cómo afecta el momento del mes en el que se solicita la baja de un familiar?",
    "¿Qué significa \"régimen de carencias médicas - habilitación progresiva\"?",
    "¿Qué ocurre si no se presenta la renovación de una afiliación sin aporte adicional?",
    "¿Qué tipo de vínculo se necesita acreditar para afiliar a un abuelo?",
    "¿Es posible afiliar a un familiar sin obra social sin realizar aporte adicional?",
    "¿Qué contacto se sugiere para casos particulares de credencial digital?",
    "¿Qué nivel de cuenta es necesario para subir documentación escaneada?",
    "¿Qué documento se solicita para probar un divorcio en trámites de afiliación?",
    "¿Se puede acceder a los trámites desde cualquier navegador o requiere app específica?",
    "¿Qué significa que un trámite es \"personal e intransferible\"?",
    "¿Qué beneficio otorga el acceso a la credencial digital de un familiar?",
    "¿Cuántas veces debe subirse la misma documentación en el trámite de asignación de representante?",
    "¿Qué tipo de escaneo se requiere para los documentos solicitados?"
]

def ask_question(question: str, api_url: str) -> Dict[str, Any]:
    """Send a question to the /ask endpoint and return the response."""
    try:
        response = requests.post(
            f"{api_url}/ask",
            json={"message": question},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "answer": None,
            "sources": []
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test questions against the RAG API')
    parser.add_argument('--output', type=str, default='test/results/questions_results.json',
                      help='Output JSON file path')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000',
                      help='API URL')
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize results list
    results = []
    
    # Process each question
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\nProcessing question {i}/{len(QUESTIONS)}: {question}")
        
        # Ask the question
        response = ask_question(question, args.api_url)
        
        # Create result entry
        result = {
            "question": question,
            "answer": response.get("answer"),
            "sources": response.get("sources", []),
            "error": response.get("error")
        }
        
        results.append(result)
        
        # Save intermediate results after each question
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Add a small delay between requests to avoid overwhelming the server
        if i < len(QUESTIONS):
            time.sleep(1)
    
    print(f"\nAll questions processed. Results saved to {output_path}")

if __name__ == "__main__":
    main() 