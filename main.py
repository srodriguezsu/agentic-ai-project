from langchain_core.messages import HumanMessage
from agent.agent import create_identity_agent
import pprint
import json
import re

agent = create_identity_agent()


def format_agent_response(response) -> str:
    """Formatea la respuesta devuelta por el agente para imprimir de forma legible.

    Intenta extraer texto de:
    - response.content (si es str)
    - response dict con 'messages' (tool outputs, AI/Human messages)
    - fallback a pprint de la estructura.
    """
    # Caso: objeto con atributo 'content'
    try:
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content.strip()
        if content is not None:
            # si es lista/dict, formatear
            return pprint.pformat(content)
    except Exception:
        pass

    # Si es dict-like con 'messages'
    if isinstance(response, dict):
        msgs = response.get("messages") or response.get("runs") or None
        if isinstance(msgs, list):
            parts = []
            for m in msgs:
                # si es dict (ToolMessage, AIMessage, HumanMessage)
                if isinstance(m, dict):
                    # ToolMessage con 'name'
                    if m.get("name") and m.get("content"):
                        parts.append(f"[TOOL: {m.get('name')}] {m.get('content')}")
                    # Mensajes con 'content' campo (string o dict)
                    elif m.get("content"):
                        parts.append(
                            m.get("content") if isinstance(m.get("content"), str) else pprint.pformat(m.get("content")))
                    else:
                        parts.append(pprint.pformat(m))
                else:
                    parts.append(str(m))
            return "\n\n".join(parts).strip()
        # Si tiene perfil final directo
        if "perfil_final" in response:
            return response["perfil_final"]
        # fallback
        return pprint.pformat(response)

    # Fallback general
    try:
        return str(response)
    except Exception:
        return "<respuesta no imprimible>"


def asks_to_save(text: str) -> bool:
    """Detecta si el agente está preguntando al usuario si desea guardar el perfil."""
    if not text:
        return False
    lower = text.lower()
    # frases comunes que indican pregunta de guardado
    patterns = [
        r"¿.*guardar.*perfil.*\?",
        r"guardar este perfil",
        r"desea guardar este perfil",
        r"quieres guardar",
        r"\bsí/no\b",
    ]
    for p in patterns:
        if re.search(p, lower):
            return True
    return False


# Mensaje inicial para generar la identidad ficticia
initial = HumanMessage(content="Genera una identidad ficticia completa y muéstramela.")

# Primera invocación
respuesta = agent.invoke({"messages": [initial]})

# Formatear y mostrar la respuesta del agente de forma legible
assistant_text = format_agent_response(respuesta)
print("Agente:")
print(assistant_text)
print("=" * 40)

# Si el agente pide explícitamente guardar, el usuario puede responder de inmediato.
if asks_to_save(assistant_text):
    user_input = input("Usuario (sí/no): ").strip()
    user_msg = HumanMessage(content=user_input)
    followup = agent.invoke({"messages": [user_msg]})
    followup_text = format_agent_response(followup)
    print("Agente:")
    print(followup_text)
    print("=" * 40)

# Entrar en REPL conversacional: el usuario puede seguir conversando con el agente
print("Puede continuar conversando con el agente. Escriba 'salir' o 'exit' para terminar.")
while True:
    user_input = input("Usuario: ").strip()
    if user_input.lower() in ("salir", "exit"):
        print("Sesión finalizada.")
        break
    if not user_input:
        continue
    user_msg = HumanMessage(content=user_input)
    respuesta = agent.invoke({"messages": [user_msg]})
    assistant_text = format_agent_response(respuesta)
    print("Agente:")
    print(assistant_text)
    print("=" * 40)
