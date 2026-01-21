import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import llm

# import des Agents
from Agent.AssistantTeacher.AssistantTeacher import AssistantTeacher
from Agent.info_UVSQ.info_UVSQ import info_UVSQ
from Agent.SmartPlanner.SmartPlanner import SmartPlanner
from Agent.RouterAgent import RouterAgent

from dotenv import load_dotenv
from typing import List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging
load_dotenv()


class GlobalStateManager:
    @staticmethod
    def init_state():
        cl.user_session.set("GlobalState", {
            "messages": [],
            "selected_agent": None
        })

    @staticmethod
    def get_state() -> Dict:
        return cl.user_session.get("GlobalState", {
            "messages": [],
            "selected_agent": None
        })

    @staticmethod
    def add_exchange(user_input: str, assistant_output: str, agent_name: str):
        state = GlobalStateManager.get_state()

        messages: List[BaseMessage] = state.get("messages", [])
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=assistant_output))

        state["messages"] = messages
        state["selected_agent"] = agent_name

        cl.user_session.set("GlobalState", state)

    @staticmethod
    def get_history() -> List[BaseMessage]:
        return GlobalStateManager.get_state().get("messages", [])

       
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("RouterAgent", RouterAgent())
    cl.user_session.set("SmartPlanner", SmartPlanner())
    cl.user_session.set("AssistantTeacher", AssistantTeacher())
    cl.user_session.set("info_UVSQ", info_UVSQ())
    GlobalStateManager.init_state()

@cl.on_message
async def handle_message(message: cl.Message):
    router = cl.user_session.get("RouterAgent")
    query = message.content
    # Route vers le bon agent
    intent = await router.ask_router(query)
    
    # Vers AssistantTeacher
    if intent == "cours":
        agent = cl.user_session.get("AssistantTeacher", AssistantTeacher())
        logging.info("\n\n>>> Agent AssistantTeacher selected <<<")
        result = await agent.ask_AssistantTeacher(query)
        last_message = result["messages"][-1].content
        # Met à jour le GlobalState
        GlobalStateManager.add_exchange(query, last_message, "AssistantTeacher")
        # renvoie la réponse
        await cl.Message(content=last_message).send()

    # Vers SmartPlanner
    elif intent == "emploi_du_temps":
        agent = cl.user_session.get("SmartPlanner", SmartPlanner())
        logging.info("\n\n>>> Agent SmartPlanner selected <<<")
        result = await agent.ask_SmartPlanner(query)
        answer = result["query_result"]
        # Met à jour le GlobalState
        GlobalStateManager.add_exchange(query, answer, "SmartPlanner")
        await cl.Message(content=answer).send()

    # Vers info_UVSQ
    elif intent == "UVSQ":
        agent = cl.user_session.get("info_UVSQ", info_UVSQ())
        logging.info("\n\n>>> Agent info_UVSQ selected <<<")
        result = await agent.ask_info_UVSQ(query)
        last_message = result["messages"][-1].content
        # Met à jour le GlobalState
        GlobalStateManager.add_exchange(query, last_message, "info_UVSQ")
        # renvoie la réponse
        await cl.Message(content=last_message).send()

    # Si la requête ne nécessite pas d'appel à un agent redirige directement vers le llm
    else:
        logging.info("\n\n>>> No agent selected <<<")
        system_prompt = """
        Tu es un assistant sympathique, plein d'esprit et avec une légère touche d'humour.
        Ton domaine d'expertise est centré sur l'Université de Versailles Saint-Quentin-en-Yvelines (UVSQ) : ses cours, ses emplois du temps, ses filières, ses services, sa vie étudiante, ou toute autre information utile aux étudiants.
        Tu es là pour aider, même quand l'utilisateur ne pose pas directement une question sur l'UVSQ.
        S'il te parle d'autre chose ou te pose une question hors sujet, réponds avec bienveillance, humour et légèreté, puis redirige subtilement la conversation en lui proposant ton aide sur des sujets liés à l'université.
        N'hésite pas à relancer avec des suggestions utiles : "Tu veux que je t'aide à retrouver ton emploi du temps ?", "Besoin d'infos sur ta filière ?", ou "Je peux aussi te filer un coup de main pour comprendre le calendrier universitaire."
        """
        hs_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", f"Question originale: {query}"),
            ]
        )
        irrelevant_response = hs_prompt | llm | StrOutputParser()
        hs_message = irrelevant_response.invoke({})
        GlobalStateManager.add_exchange(query, hs_message, "none")
        await cl.Message(content=hs_message).send()

    # agent = cl.user_session.get("AssistantTeacher")
    # # Envoie la requête utilisateur à l'agent
    # result = await agent.ask_AssistantTeacher(query=message.content)
    # # Récupère la dernière réponse de l'assistant dans le state
    # await cl.Message(content=result).send()