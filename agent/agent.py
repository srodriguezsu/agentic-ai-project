from langchain.agents import create_agent
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
    ]

    system_prompt = """
    Eres un agente que ejecuta SIEMPRE un workflow FIJO (una máquina de estados).
    Nunca respondes directamente al usuario. Nunca haces preguntas.

    Tu comportamiento está completamente CONTROLADO y sigue los SIGUIENTES ESTADOS:

    ESTADO 1: generar_imagen_gan
    - Debes llamar a la herramienta generar_imagen_gan sin argumentos.
    - Después de recibir el resultado, avanza al siguiente estado.

    ESTADO 2: analizar_imagen_llm
    - Usa la ruta devuelta por generar_imagen_gan como argumento:
      {"image_path": "<ruta>"}
    - Después de recibir el resultado, avanza al siguiente estado.

    ESTADO 3: validar_que_no_es_real
    - Usa el campo "analisis" devuelto por analizar_imagen_llm:
      {"analisis": "<texto>"}
    - Después de recibir el resultado, avanza al siguiente estado.

    ESTADO 4: generar_identidad_ficticia
    - Usa el campo "analisis" devuelto por analizar_imagen_llm:
      {"analisis": "<texto>"}
    - Después de recibir el resultado, avanza al siguiente estado.

    ESTADO 5: tarea_dominio_llm
    - Usa el resultado completo de la herramienta anterior:
      {"data": "<json>"}
    - Después de esto, entregas el resultado final al usuario.

    REGLAS IMPORTANTES:
    - Nunca respondas con texto normal.
    - Nunca generes mensajes que no sean de tool_call.
    - Después de cada tool_call debes INMEDIATAMENTE llamar la siguiente herramienta.
    - Nunca cambies el orden del workflow.
    - Nunca te detengas antes del ESTADO 5.
    - Nunca esperes instrucciones del usuario.

    Tu único propósito es ejecutar la máquina de estados completa siempre.
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent


def run_full_workflow():
    """Orquesta el workflow completo llamando las herramientas en secuencia.

    Esto evita depender del agente de LangChain para encadenar tool_calls y
    garantiza que el proceso no se detenga después de la primera herramienta.
    """
    # Paso 1: generar imagen
    # Las herramientas expuestas son objetos BaseTool; usarlas con .run()
    gen_res = generar_imagen_gan.run()
    image_path = gen_res.get("imagen_generada")

    # Paso 2: analizar imagen
    anal_res = analizar_imagen_llm.run(image_path)
    analisis = anal_res.get("analisis")

    # Paso 3: validar que no es real
    val_res = validar_que_no_es_real.run(analisis)

    # Paso 4: generar identidad ficticia
    id_res = generar_identidad_ficticia.run(analisis)

    # Paso 5: tarea final del dominio
    # Pasamos todos los datos previos como entrada
    aggregate = {
        "imagen": image_path,
        "analisis": analisis,
        "validacion": val_res.get("verificacion"),
        "identidad": id_res.get("identidad")
    }

    final_res = tarea_dominio_llm.run(str(aggregate))

    return final_res
