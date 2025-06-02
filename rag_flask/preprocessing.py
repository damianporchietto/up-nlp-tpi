import re
import unicodedata
import nltk
from nltk.corpus import stopwords

# Asegura que los recursos se encuentren disponibles
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

spanish_stopwords = set(stopwords.words('spanish'))


def normalize_text(text: str) -> str:
    """
    Aplica limpieza básica: minúsculas, quitar tildes, caracteres especiales, espacios, etc.
    """
    if not isinstance(text, str):
        return ""

    # Minssculas
    text = text.lower()

    # Quitar tildes y acentos
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # Quitar caracteres especiales (mantener letras, numeros y espacios)
    text = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", text)

    # Eliminar multiples espacios
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_stopwords(text: str) -> str:
    """
    Elimina stopwords
    """
    tokens = text.split()
    filtered = [word for word in tokens if word not in spanish_stopwords]
    return " ".join(filtered)


def clean_and_tokenize(text: str) -> str:
    """
    Aplicar limpieza + remoción de stopwords
    """
    text = normalize_text(text)
    text = remove_stopwords(text)
    return text


def flatten_list_field(lst):
    """
    Convierte listas (de pasos o requisitos) en un string limpio.
    Se puede obviar si es necesario
    """
    if not isinstance(lst, list):
        return ""
    return " - " + " - ".join(clean_and_tokenize(item) for item in lst if isinstance(item, str))


def preprocess_document(doc: dict) -> dict:
    """
    Preprocesa un documento JSON: limpia texto, normaliza, elimina stopwords,
    convierte listas en texto plano y genera campo unificado 'content'.
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
