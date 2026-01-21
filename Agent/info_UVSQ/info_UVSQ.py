from langchain_core.messages import AnyMessage, HumanMessage
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from utils.utils import llm
from langchain_core.messages import SystemMessage
import functools
import chromadb
import logging

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

@tool
def retrieve_context(query: str):
    """Utilise les contextes les plus liés à la question de l'utilisateur pour y répondre. Ces contextes sont extraits du site de l'UVSQ."""
    ef = SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")
    collection_path = r"Agent/info_UVSQ/DBv"
    collection_name = "UVSQ_DOCS"
    client = chromadb.PersistentClient(path=collection_path)
    collection = client.get_collection(name=collection_name, embedding_function=ef)
    contexts = collection.query(
        query_texts=[query],
        n_results=3
        )
    # log les contextes
    context_list = contexts['documents'][0]
    formatted_context = "\n\n".join(
        f"Contexte {i + 1} :\n{doc}" for i, doc in enumerate(context_list)
    )
    logging.info(f"[CONTEXT] pour query='{query}' :\n{formatted_context}")
    return formatted_context




# ---- State
class StateUVSQ(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ---- Agent
class info_UVSQ():

    def __init__(self):
        
        self.system_prompt = """
        Tu es un assistant conçu et spécialisé pour répondre aux informations sur le site de l'UVSQ (l'Université Versailles Saint-Quentin en Yvelines).
        Pour toute question, tu dois OBLIGATOIREMENT appeler le tool `retrieve_context` AVANT de formuler une réponse.

        NE réponds PAS sans avoir appelé ce tool.
        Tu ne dois pas deviner la réponse toi-même sans contexte. Ce tool te retourne des documents pertinents extraits du site, que tu dois ensuite utiliser pour formuler une réponse bien argumentée.

        Format de ton raisonnement :
        1. Appelle `retrieve_context` avec la question posée.
        2. Utilise les extraits des pages du site retournés pour construire une réponse structurée, pédagogique et exacte.
        """
        
        tools = [retrieve_context]
        self.tooled_llm = llm.bind_tools(tools)
        
        builder = StateGraph(StateUVSQ)
        builder.add_node("tool_calling_llm", functools.partial(self.tool_calling_llm))
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "tool_calling_llm")
        builder.add_conditional_edges("tool_calling_llm", tools_condition)
        builder.add_edge("tools", "tool_calling_llm")

        self.graph = builder.compile()

    async def tool_calling_llm(self, state: StateUVSQ) -> StateUVSQ:
        result = await self.tooled_llm.ainvoke(state["messages"])
        return {"messages": state["messages"] + [result]}

    async def ask_info_UVSQ(self, query: str):
        response = await self.graph.ainvoke({"messages": [SystemMessage(content=self.system_prompt), HumanMessage(content=query)]})
        return response


# ----------- TEST -----------
# import asyncio
# import logging

# agent = info_UVSQ()

# query_1 = "Quelle place occupe l'UVSQ dans le classement CWUR 2024 ?  ²"
# query_2 = "Quelles formations de master reconnues internationalement propose l'UVSQ ?"
# query_3 = "Quel est le rôle de l'UVSQ au sein de l'Université Paris-Saclay ?"
# query_4 = "Qu'est-ce qui a été mis en avant lors de l'édition d'avril 2022 dans la revue « Décisions Achats » ?"
# query_5  = "Quels types d'espaces sont accessibles aux étudiants pour se détendre ou travailler sur les campus de l’UVSQ ?"
# query_6  = "Quelles salles peut-on réserver à la Maison de l’Étudiant Marta Pan à Saint-Quentin-en-Yvelines ?"
# query_7  = "Où se trouve la Maison de l'Étudiant Marta Pan et à quoi sert-elle ?"

# query_8  = "Quels partenaires proposent des tarifs réduits aux étudiants de l’UVSQ ?"
# query_9  = "L’UVSQ propose-t-elle des aides pour accéder à des salles de sport ou des activités de loisir ?"
# query_10 = "Existe-t-il un partenariat entre l’UVSQ et des clubs de tennis ou de fitness ?"

# query_11 = "Que propose l’UVSQ en cas de règles menstruelles douloureuses ?"
# query_12 = "Le port du masque est-il encore obligatoire à l’UVSQ ?"
# query_13 = "Qui peut-on contacter en cas de détresse psychologique sur un campus de l’UVSQ ?"
# query_14 = "Que faire si je me blesse pendant un stage ou sur le campus ?"

# query_15 = "Quels aménagements sont proposés par l’UVSQ pour les examens des étudiants en situation de handicap ?"
# query_16 = "Est-il possible d’obtenir une transcription en braille ou l’assistance d’un interprète LSF à l’UVSQ ?"
# query_17 = "Le service handicap de l’UVSQ peut-il fournir un secrétaire pour rédiger les examens ?"

# query_18 = "Quels sont les numéros d'urgence disponibles pour les victimes de harcèlement ou de violences à l’UVSQ ?"
# query_19 = "Quelles associations partenaires peuvent accompagner les victimes de violences dans les Yvelines ?"
# query_20 = "À qui s’adresser en cas de harcèlement sexuel à l’université ?"
# query_21 = "Quels dispositifs sont mis en place par l’UVSQ pour lutter contre les discriminations et agissements sexistes ?"

# query_22 = "Qu’est-ce que le dispositif Culture-ActionS du CROUS et à quoi sert-il ?"
# query_23 = "Que propose le service culturel de l’UVSQ aux étudiants et personnels ?"
# query_24 = "Peut-on accueillir des artistes ou organiser des spectacles sur les campus ?"
# query_25 = "En quoi consiste l’UE Engagement proposée par l’UVSQ et combien de crédits ECTS permet-elle de valider ?"

# query_26 = "Quels services de restauration sont disponibles sur le campus de Mantes ?"
# query_27 = "Y a-t-il des cafétérias tenues par des associations à l’UVSQ ?"
# query_28 = "Peut-on accéder à des plats chauds ou des sandwichs sur les différents campus ?"

# query_29 = "Quand peut-on faire une demande de logement ou de bourse via le CROUS ?"
# query_30 = "Est-il possible de consulter gratuitement des annonces de logements étudiants via l’UVSQ ?"

# query_31 = "Quels types de pratiques sportives sont proposées par le SUAPS de l’UVSQ ?"
# query_32 = "Existe-t-il des dispositifs sportifs adaptés pour les étudiants en situation de handicap ?"
# query_33 = "Comment s’engage l’UVSQ pour le sport-santé ?"
# query_34 = "Quels avantages ou tarifs préférentiels sont proposés pour les activités sportives à l’UVSQ ?"

# query_35 = "Où peut-on consulter les offres d’emploi ou de stage liées à l’UVSQ ?"
# query_36 = "Quelles entreprises proposent des stages ou des alternances en partenariat avec l’UVSQ ?"
# query_37 = "Est-ce que le réseau Alumni de l’UVSQ diffuse régulièrement des offres ?"

# result = asyncio.run(agent.ask_info_UVSQ(query_2))

# result["messages"][-1].content

