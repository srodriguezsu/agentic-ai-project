from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.tools import (
    generar_imagen_gan,
    analizar_imagen_llm,
    validar_que_no_es_real,
    generar_identidad_ficticia,
    tarea_dominio_llm,
)
from agent.config import GEMINI_API_KEY


def create_identity_agent():
    """
    Crea el agente de identidades ficticias usando la API moderna de LangChain,
    con Gemini como el modelo principal para el reasoning general del agente.
    """

    # Gemini será el modelo principal del agente (razonamiento + selección de tools)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    tools = [
        generar_imagen_gan,
        analizar_imagen_llm,
        validar_que_no_es_real,
        generar_identidad_ficticia,
        tarea_dominio_llm,
    ]

    system_prompt = """
    Eres un agente especializado en crear identidades humanas completamente ficticias.
    Tienes acceso a varias herramientas que te permiten:

    - Generar retratos sintéticos mediante una GAN entrenada.
    - Analizar retratos visualmente (edad, género, emoción, estilo).
    - Validar que el retrato NO pertenezca a una persona real.
    - Generar identidades ficticias completas basadas en el análisis.
    - Producir un perfil final coherente y listo para usar.

    Usa las herramientas cuando sea necesario.
    Si no necesitas ninguna herramienta, responde directamente.
    Siempre prioriza el uso de JSON cuando corresponda.
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent
