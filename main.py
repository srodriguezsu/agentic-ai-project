from agent.agent import create_identity_agent
from langchain_core.messages import HumanMessage

agent = create_identity_agent()

respuesta = agent.invoke({
    "messages": [
        HumanMessage(content="Genera una identidad ficticia completa")
    ]
})

print(respuesta)
