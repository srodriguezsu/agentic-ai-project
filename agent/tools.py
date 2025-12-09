import base64
import mimetypes

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
    path = generate_portrait()
    return {"imagen_generada": path}


# -----------------------------
# TOOL 2: Análisis visual (Gemini)
# -----------------------------
@tool("analizar_imagen_llm", return_direct=False)
def analizar_imagen_llm(image_path: str):
    """Analiza atributos visuales del retrato (edad, género, emoción) usando Gemini Vision."""

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
                    "- estilo visual"
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

    result = groq_chat(prompt)
    return {"perfil_final": result}
