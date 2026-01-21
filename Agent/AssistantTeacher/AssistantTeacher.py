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
    """Utilise les contextes les plus liés à la question de l'utilisateur pour y répondre. Ces contextes sont extraits des cours de l'université vectorisés."""
    ef = SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")
    collection_path = r"Agent\AssistantTeacher\DBv"
    collection_name = "STAT_NON_PARAM_v"
    client = chromadb.PersistentClient(path=collection_path)
    collection = client.get_collection(name=collection_name, embedding_function=ef)
    contexts = collection.query(
            query_texts=[query],
            n_results=1
            )
    # Ajoute les entêtes "Contexte 1:", etc.
    context_list = contexts['documents'][0]
    formatted_context = "\n\n".join(
        f"Contexte {i + 1} :\n{doc}" for i, doc in enumerate(context_list)
    )
    logging.info(f"[CONTEXT] pour query='{query}' :\n{formatted_context}")
    return formatted_context


# ---- State
class StateTeacher(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ---- Agent
class AssistantTeacher():

    def __init__(self):
        
        self.system_prompt = r"""
        Tu es un assistant spécialisé dans l'enseignement des statistiques.
        Pour toute question posée, tu dois OBLIGATOIREMENT commencer par appeler le tool retrieve_context avec la question exacte, afin de récupérer les extraits pertinents des cours.

        NE donne AUCUNE réponse avant d'avoir utilisé retrieve_context.

        Une fois les extraits récupérés, utilise-les pour formuler une réponse pédagogique, précise, bien structurée et rigoureuse.
        Ces contextes peuvent contenir des formules mathématiques, écrites en LaTeX entre '$' ou '$$'. Lorsque tu cites ces formules garde ce format LaTeX.
        Tu dois IMPÉRATIVEMENT écrire toutes les formules mathématiques et variables en LaTeX y compris les notations simples, comme par exemple $X_i$, $f(x)$, ou $$P(A \cap B) = P(A) \cdot P(B|A)$$.

        Ton raisonnement doit suivre ce schéma :
            - Appel au tool retrieve_context.
            - Formulation claire et structurée de la réponse avec toutes les expressions mathématiques encadrées de $$ pour un rendu LaTeX.
        Ne reformule pas ou n'invente pas les concepts si tu ne les trouves pas dans les extraits retournés. Dis simplement que l'information est absente du contexte.
        """
        
        tools = [retrieve_context]
        self.tooled_llm = llm.bind_tools(tools)
        
        builder = StateGraph(StateTeacher)
        builder.add_node("tool_calling_llm", functools.partial(self.tool_calling_llm))
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "tool_calling_llm")
        builder.add_conditional_edges("tool_calling_llm", tools_condition)
        builder.add_edge("tools", "tool_calling_llm")

        self.graph = builder.compile()

    async def tool_calling_llm(self, state: StateTeacher) -> StateTeacher:
        result = await self.tooled_llm.ainvoke(state["messages"])
        return {"messages": state["messages"] + [result]}

    async def ask_AssistantTeacher(self, query: str):
        response = await self.graph.ainvoke({"messages": [SystemMessage(content=self.system_prompt), HumanMessage(content=query)]})
        return response

# ---- TESTS ----
queries = [
    "Quelle est la différence fondamentale entre probabilité et statistique ?",
    "Pourquoi dit-on qu'un problème est non paramétrique ?",
    "Quel est l'avantage de ne pas imposer de modèle sur la forme de la densité ?",
    "Quelles sont les étapes pour construire un estimateur par histogramme ?",
    "Pourquoi considère-t-on une partition uniforme de [0,1] ?",
    "En quoi l'estimation par histogramme transforme un problème non paramétrique en un problème paramétrique ?",
    "Quelle est la signification du paramètre h dans l'histogramme ?",
    "Comment détermine-t-on les constantes $p_j$ dans chaque classe $C_j$ ?",
    "Pourquoi l'estimateur par histogramme est-il une densité ?",
    "Qu'est-ce que la convergence forte des estimateurs ?",
    "Que représente l'erreur quadratique moyenne (MISE) ?",
    "Comment décompose-t-on le MISE en biais et variance ?",
    "Dans quelles conditions $\\( \\hat{f}_h \\)$ devient-il un estimateur fortement consistant ?",
    "Que se passe-t-il lorsque h → 0 et n → ∞ ?",
    "Comment interpréter les termes $f'(x)^2$ dans le risque quadratique moyen ?",
    "Pourquoi le choix théorique optimal de h n'est-il pas réalisable en pratique ?",
    "Comment la validation croisée permet-elle de déterminer h de façon empirique ?",
    "Dans quel type de données appliquerait-on un estimateur par histogramme ?",
    "Quels sont les risques d'un mauvais choix de h ?",
    "Quelle méthode pourrait être plus adaptée que l'histogramme pour une estimation de densité lisse ?"
]