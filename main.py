from langchain_core.messages import HumanMessage
from agent.agent import create_identity_agent, run_full_workflow

agent = create_identity_agent()

respuesta = run_full_workflow()

print(respuesta)
