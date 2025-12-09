from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
from agent.config import GEMINI_API_KEY, GROQ_API_KEY
from gan.generate import generate_portrait

# LLM de visión: Gemini
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.3
)

# Cliente oficial de Groq
groq = Groq(api_key=GROQ_API_KEY)


def groq_chat(prompt: str, model: str = "llama3-70b-8192") -> str:
    """Función auxiliar para enviar prompts al modelo Groq."""
    response = groq.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message["content"]


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
    """Analiza atributos visuales del retrato (edad, género, emoción)."""
    # Abrimos la imagen solo para validar que existe, pero no la enviamos
    # como dict tipo 'image' porque la interfaz de mensajes espera 'role'/'content'.
    with open(image_path, "rb") as f:
        _ = f.read()

    prompt = (
        f"Analiza este retrato localizado en la ruta: {image_path}. "
        "Describe edad aparente, género percibido, emoción y estilo visual. "
        "Responde con un texto claro y conciso."
    )

    # Enviar mensaje con las claves 'role' y 'content' para evitar el error de coerción
    result = gemini.invoke([
        {"role": "user", "content": prompt}
    ])

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
