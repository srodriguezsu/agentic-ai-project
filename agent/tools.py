from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
from agent.config import GEMINI_API_KEY, GROQ_API_KEY
from gan.generate import generate_portrait

# LLM de visión: Gemini
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=GEMINI_API_KEY,
    temperature=0.3
)

# LLM de razonamiento: Groq (Llama 3)
groq = Groq(api_key=GROQ_API_KEY)


# -----------------------------
# TOOL 1: Generar retrato (GAN)
# -----------------------------
@tool("generar_imagen_gan", return_direct=True)
def generar_imagen_gan(descripcion: str = ""):
    """Genera un retrato sintético usando la GAN entrenada."""
    path = generate_portrait(descripcion)
    return {"imagen_generada": path}


# -----------------------------
# TOOL 2: Análisis visual (Gemini)
# -----------------------------
@tool("analizar_imagen_llm", return_direct=True)
def analizar_imagen_llm(image_path: str):
    """Analiza atributos visuales del retrato (edad, género, emoción)."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = gemini.invoke([
        {"type": "text",
         "text": "Analiza este retrato y describe edad aparente, género percibido, emoción y estilo visual."},
        {"type": "image", "image": image_bytes}
    ])

    return {"analisis": result.content}


# -----------------------------
# TOOL 3: Validar que no es real
# -----------------------------
@tool("validar_que_no_es_real", return_direct=True)
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
    result = groq.invoke(prompt)
    return {"verificacion": result}


# -----------------------------
# TOOL 4: Generar identidad ficticia
# -----------------------------
@tool("generar_identidad_ficticia", return_direct=True)
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

    result = groq.invoke(prompt)
    return {"identidad": result}


# -----------------------------
# TOOL 5: Tarea final del dominio
# -----------------------------
@tool("tarea_dominio_llm", return_direct=True)
def tarea_dominio_llm(data: str):
    """Integra todos los datos y produce el perfil final."""
    prompt = f"""
    Construye un perfil final listo para usar en pruebas o demos.
    Entrada:

    {data}

    Entrega un JSON cohesivo con toda la información.
    """

    result = groq.invoke(prompt)
    return {"perfil_final": result}
