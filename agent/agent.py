from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.tools import (
    generar_imagen_gan,
    analizar_imagen_llm,
    validar_que_no_es_real,
    generar_identidad_ficticia,
    tarea_dominio_llm,
    guardar_perfil_csv,  # agregada la nueva herramienta
)
from agent.config import GEMINI_API_KEY
import json


def create_identity_agent():
    """
    Crea el agente de identidades ficticias en modo conversacional,
    consciente de las herramientas y preguntando al usuario antes de guardar en CSV.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    tools = [
        generar_imagen_gan,
        analizar_imagen_llm,
        validar_que_no_es_real,
        generar_identidad_ficticia,
        tarea_dominio_llm,
        guardar_perfil_csv,
    ]

    # Prompt del sistema: más conversacional y explícito sobre comportamiento y guardado.
    system_prompt = """
    Eres un asistente conversacional amable y eficiente que genera perfiles ficticios usando herramientas.
    Prioriza la claridad y la interacción con el usuario. Sigue estas pautas:

    - Puedes hacer preguntas aclaratorias antes de ejecutar las herramientas si algo no está claro.
    - Explica brevemente cada paso cuando sea relevante (por ejemplo: "Voy a generar una imagen sintética ahora").
    - Ejecuta las herramientas cuando sea necesario para completar la tarea en el siguiente orden si procede:
        1) generar_imagen_gan
        2) analizar_imagen_llm (usar la ruta devuelta por la tool anterior)
        3) validar_que_no_es_real (usar el análisis)
        4) generar_identidad_ficticia (usar el análisis)
        5) tarea_dominio_llm (construir el perfil final)

    - Tras producir el perfil final: MUESTRA el perfil al usuario en formato JSON y PREGUNTA explícitamente:
      "¿Desea guardar este perfil en el CSV? (sí/no)"
    - NO llames a la herramienta guardar_perfil_csv sin confirmación explícita del usuario.
    - Si el usuario responde "sí", llama a guardar_perfil_csv con el JSON del perfil.
    - Si el usuario responde "no", confirma que no se guardará y ofrece acciones alternativas (editar, regenerar, exportar).
    - Mantén respuestas cortas, amables y orientadas a la acción.

    Si el usuario desea editar campos del perfil antes de guardar, guía el proceso de edición (pregunta qué campo editar, recibe el nuevo valor y aplica el cambio).
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent
