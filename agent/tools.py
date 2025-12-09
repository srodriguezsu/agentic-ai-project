import base64
import mimetypes
import csv
import json
from datetime import datetime

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from groq import Groq
from agent.config import GEMINI_API_KEY, GROQ_API_KEY
from gan.generate import generate_portrait
import os

# LLM de visión: Gemini
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.3
)

# Cliente oficial de Groq
groq = Groq(api_key=GROQ_API_KEY)


def groq_chat(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 512) -> str:
    """Función auxiliar para enviar prompts al modelo Groq con límite de tokens y manejo de error."""
    response = groq.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# -----------------------------
# TOOL 1: Generar retrato (GAN)
# -----------------------------
@tool("generar_imagen_gan", return_direct=False)
def generar_imagen_gan():
    """Genera un retrato sintético usando la GAN entrenada."""
    print('Generando retrato sintético...')
    path = generate_portrait()
    return {"imagen_generada": path}


# -----------------------------
# TOOL 2: Análisis visual (Gemini)
# -----------------------------
@tool("analizar_imagen_llm", return_direct=False)
def analizar_imagen_llm(image_path: str):
    """Analiza atributos visuales del retrato (edad, género, emoción) usando Gemini Vision."""
    print("Analizando imagen...")

    if not os.path.exists(image_path):
        return {"error": "ruta de imagen no encontrada"}

    image_bytes = open(image_path, "rb").read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analiza este retrato y describe brevemente:\n"
                    "- edad aparente\n"
                    "- género percibido\n"
                    "- emoción principal\n"
                    "- características distintivas\n"
                    "- cualquier otro detalle relevante\n"
                )
            },
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": mime,
            },
        ]
    )

    result = gemini.invoke([message])

    return {"analisis": result.content}


# -----------------------------
# TOOL 3: Validar que no es real
# -----------------------------
@tool("validar_que_no_es_real", return_direct=False)
def validar_que_no_es_real(analisis: str):
    """Verifica que el retrato no coincida con una persona real."""
    print("Validando que el retrato no coincida con una persona real...")
    prompt = f"""
    Basado en este análisis visual:

    {analisis}

    Evalúa si este retrato podría coincidir con una persona real o celebridad.
    Responde en JSON:
    {{
        "seguro": true/false,
        "explicacion": "texto"
    }}
    """

    result = groq_chat(prompt)
    return {"verificacion": result}


# -----------------------------
# TOOL 4: Generar identidad ficticia
# -----------------------------
@tool("generar_identidad_ficticia", return_direct=False)
def generar_identidad_ficticia(analisis: str):
    """Genera nombre, biografía, personalidad y datos ficticios."""
    print("Generando identidad ficticia...")
    prompt = f"""
    Con base en este análisis del rostro:

    {analisis}

    Genera una identidad completamente ficticia.
    Formato JSON:
    {{
        "nombre": "...",
        "apellido": "...",
        "edad_aproximada": ...,
        "genero_percibido": "...",
        "ocupacion": "...",
        "personalidad": "...",
        "hobbies": ["..."],
        "biografia": "..."
    }}
    """

    result = groq_chat(prompt)
    return {"identidad": result}


# -----------------------------
# TOOL 5: Tarea final del dominio
# -----------------------------
@tool("tarea_dominio_llm", return_direct=False)
def tarea_dominio_llm(data: str):
    """Integra todos los datos y produce el perfil final."""
    prompt = f"""
    Construye un perfil final listo para usar en pruebas o demos.
    Entrada:

    {data}

    Entrega un JSON cohesivo con toda la información.
    """
    print("Tarea dominio llm...")

    result = groq_chat(prompt)
    return {"perfil_final": result}


# -----------------------------
# TOOL 6: Guardar perfil en CSV
# -----------------------------
@tool("guardar_perfil_csv", return_direct=False)
def guardar_perfil_csv(perfil: str = "", csv_path: str = "profiles.csv"):
    """Guarda un perfil (JSON string o dict) en un CSV local.

    Args:
        perfil: JSON string (preferible) o texto plano con la estructura generada por la herramienta.
                Se declara como str para que la librería pueda generar correctamente el esquema.
        csv_path: ruta del CSV donde se almacenarán los perfiles.

    Devuelve:
        dict con status, csv_path y message (texto amigable).
    """
    print("Guardando perfil...")
    # Si no se proporciona perfil, devolver un error claro (evita None en el esquema)
    if not perfil:
        return {"status": "error", "error": "perfil vacío", "message": "No se proporcionó ningún perfil para guardar."}

    # Normalizar perfil a dict si es posible
    perfil_dict = None
    # Si la entrada parece ser ya una representación de dict (no probable aquí porque tipado como str),
    # intentamos parsearla como JSON.
    try:
        perfil_dict = json.loads(perfil)
    except Exception:
        perfil_dict = None

    # Campos previsibles para columnas
    cols = [
        "timestamp",
        "nombre",
        "apellido",
        "edad_aproximada",
        "genero_percibido",
        "ocupacion",
        "personalidad",
        "hobbies",
        "biografia",
        "raw_json"
    ]

    # Construir la fila base
    row = {c: "" for c in cols}
    row["timestamp"] = datetime.utcnow().isoformat()

    if perfil_dict:
        # Extraer campos comunes si existen
        row["nombre"] = perfil_dict.get("nombre", "")
        row["apellido"] = perfil_dict.get("apellido", "")
        row["edad_aproximada"] = perfil_dict.get("edad_aproximada", "")
        row["genero_percibido"] = perfil_dict.get("genero_percibido", "")
        row["ocupacion"] = perfil_dict.get("ocupacion", "")
        row["personalidad"] = perfil_dict.get("personalidad", "")
        hobbies = perfil_dict.get("hobbies", "")
        # asegurarse que hobbies sea string
        if isinstance(hobbies, (list, tuple)):
            row["hobbies"] = json.dumps(hobbies, ensure_ascii=False)
        else:
            row["hobbies"] = str(hobbies)
        row["biografia"] = perfil_dict.get("biografia", "")
        row["raw_json"] = json.dumps(perfil_dict, ensure_ascii=False)
    else:
        # No se pudo parsear: guardar todo en raw_json
        row["raw_json"] = str(perfil)

    # Escribir/añadir al CSV
    file_exists = os.path.exists(csv_path)
    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        return {"status": "error", "error": str(e), "message": f"No se pudo guardar el perfil: {e}"}

    return {"status": "ok", "csv_path": csv_path, "message": f"Perfil guardado en {csv_path}"}
