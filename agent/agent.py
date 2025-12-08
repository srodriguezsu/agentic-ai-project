from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from groq import Groq
from agent.tools import (
    generar_imagen_gan,
    analizar_imagen_llm,
    validar_que_no_es_real,
    generar_identidad_ficticia,
    tarea_dominio_llm,
)
from agent.config import GROQ_API_KEY


def create_identity_agent():
    """Crea el agente de identidades ficticias usando la API moderna de LangChain."""

    llm = Groq(api_key=GROQ_API_KEY)

    tools = [
        generar_imagen_gan,
        analizar_imagen_llm,
        validar_que_no_es_real,
        generar_identidad_ficticia,
        tarea_dominio_llm,
    ]

    system_prompt = """
    Eres un agente experto en la generación de identidades humanas ficticias.
    Dispones de herramientas que te permiten:
    - generar retratos sintéticos,
    - analizarlos visualmente,
    - validar que no representen a personas reales,
    - generar perfiles demográficos y biográficos,
    - entregar un perfil final completo.

    Usa las herramientas cuando sea necesario.
    Siempre entrega resultados claros, seguros y en JSON cuando sea posible.
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent
