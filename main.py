from langchain_core.messages import HumanMessage
from agent.agent import create_identity_agent

agent = create_identity_agent()

respuesta = agent.invoke({
    "messages": [
        HumanMessage(content="Genera una identidad ficticia completa")
    ]
})

print(respuesta)
