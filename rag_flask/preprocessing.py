import re
import unicodedata
# import nltk  # Removido - no necesario para Transformers
# from nltk.corpus import stopwords  # Removido - contraproducente

# Removido - no usar stopwords con Transformers
# try:
#     nltk.data.find("corpora/stopwords")
# except LookupError:
#     nltk.download("stopwords")
# spanish_stopwords = set(stopwords.words('spanish'))


def normalize_text(text: str) -> str:
    """
    Aplica limpieza básica apropiada para modelos Transformer:
    - Normaliza espacios en blanco
    - Mantiene acentos y tildes (importantes para español)
    - Mantiene stopwords (aportan contexto sintáctico)
    - Solo elimina caracteres verdaderamente problemáticos
    """
    if not isinstance(text, str):
        return ""

    # Convertir a minúsculas para consistencia
    text = text.lower()

    # MANTENER tildes y acentos - son importantes para español y Transformers
    # (Comentando la remoción que era contraproducente)
    # text = ''.join(
    #     c for c in unicodedata.normalize('NFD', text)
    #     if unicodedata.category(c) != 'Mn'
    # )

    # Eliminar solo caracteres verdaderamente problemáticos (mantener letras con acentos)
    # Patrón actualizado para preservar caracteres acentuados
    text = re.sub(r"[^\w\sáéíóúüñ¿¡]", " ", text)

    # Normalizar múltiples espacios a uno solo
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# FUNCIÓN REMOVIDA - contraproducente para Transformers
# def remove_stopwords(text: str) -> str:
#     """
#     Elimina stopwords - REMOVIDO porque es contraproducente para Transformers
#     Los modelos modernos necesitan stopwords para contexto sintáctico
#     """
#     tokens = text.split()
#     filtered = [word for word in tokens if word not in spanish_stopwords]
#     return " ".join(filtered)


def clean_and_tokenize(text: str) -> str:
    """
    Aplicar solo limpieza básica apropiada para Transformers
    - NO remover stopwords (aportan contexto)
    - NO remover tildes (importantes en español)
    """
    text = normalize_text(text)
    # text = remove_stopwords(text)  # REMOVIDO - contraproducente
    return text


def flatten_list_field(lst):
    """
    Convierte listas (de pasos o requisitos) en un string limpio.
    Mantiene formato legible para el modelo
    """
    if not isinstance(lst, list):
        return ""
    return " - " + " - ".join(clean_and_tokenize(item) for item in lst if isinstance(item, str))


def preprocess_document(doc: dict) -> dict:
    """
    Preprocesa un documento JSON con estrategia apropiada para Transformers:
    - Mantiene contexto sintáctico completo
    - Preserva acentos y caracteres importantes del español
    - Solo normaliza espacios y caracteres problemáticos
    """
    title = clean_and_tokenize(doc.get("title", ""))
    description = clean_and_tokenize(doc.get("description", ""))

    requirements_list = []
    if isinstance(doc.get("requirements"), list):
        for req in doc["requirements"]:
            if isinstance(req, dict):
                req_title = clean_and_tokenize(req.get("title", ""))
                req_content = clean_and_tokenize(req.get("content", ""))
                requirements_list.append(f"{req_title}: {req_content}")
    requirements = flatten_list_field(requirements_list)

    steps = flatten_list_field(doc.get("steps", []))

    content = f"{title}\n\nDescripcion:\n{description}\n\nRequisitos:{requirements}\n\nPasos:{steps}"

    return {
        "title": title,
        "description": description,
        "requirements": requirements,
        "steps": steps,
        "content": content,
        "url": doc.get("url", "")
    }
