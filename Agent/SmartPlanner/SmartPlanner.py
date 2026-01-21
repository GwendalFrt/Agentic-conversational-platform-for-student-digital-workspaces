from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from utils.utils import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import text
from Agent.SmartPlanner.prompts import relevance_prompt, text_to_sql_prompt
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

# === BDD setup ===
DB_PATH = r"Agent\SmartPlanner\data\edt.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SQLDatabase(engine)

class StatePlanner(TypedDict):
    question: str
    sql_query: str
    query_result: str
    query_rows: list
    attempts: int
    relevance: str
    sql_error: bool

class ConvertToSQL(BaseModel):
    sql_query: str = Field(
        description="The SQL query corresponding to the user's natural language question."
    )

class CheckRelevance(BaseModel):
    relevance: str = Field(
        description="Indique si la question est liée au schéma de la base de données. 'relevant' ou 'not_relevant'."
    )

class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")

class SmartPlanner:
    def __init__(self):
        workflow = StateGraph(StatePlanner)

        # nodes
        workflow.add_node("check_relevance", self.check_relevance)
        workflow.add_node("convert_to_sql", self.convert_nl_to_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("generate_human_readable_answer", self.generate_human_readable_answer)
        workflow.add_node("regenerate_query", self.regenerate_query)
        workflow.add_node("generate_funny_response", self.generate_funny_response)
        workflow.add_node("end_max_iterations", self.end_max_iterations)

        # edges
        workflow.add_conditional_edges(
            "check_relevance",
            self.relevance_router,
            {
                "convert_to_sql": "convert_to_sql",
                "generate_funny_response": "generate_funny_response",
            },
        )
        workflow.add_edge("convert_to_sql", "execute_sql")
        workflow.add_conditional_edges(
            "execute_sql",
            self.execute_sql_router,
            {
                "generate_human_readable_answer": "generate_human_readable_answer",
                "regenerate_query": "regenerate_query",
            },
        )
        workflow.add_conditional_edges(
            "regenerate_query",
            self.check_attempts_router,
            {
                "convert_to_sql": "convert_to_sql",
                "max_iterations": "end_max_iterations",
            },
        )
        workflow.add_edge("generate_human_readable_answer", END)
        workflow.add_edge("generate_funny_response", END)
        workflow.add_edge("end_max_iterations", END)
        workflow.set_entry_point("check_relevance")

        self.graph = workflow.compile()

    @staticmethod
    def check_relevance(state: StatePlanner):
        question = state["question"]
        print(f"Checking relevance of the question: {question}")
        human = f"Question: {question}"
        check_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", relevance_prompt),
                ("human", human),
            ]
        )
        structured_llm = llm.with_structured_output(CheckRelevance)
        relevance_checker = check_prompt | structured_llm
        relevance = relevance_checker.invoke({})
        state["relevance"] = relevance.relevance
        print(f"Relevance determined: {state['relevance']}")
        return state

    @staticmethod
    def convert_nl_to_sql(state: StatePlanner):
        question = state["question"]
        print(f"Converting question to SQL for user: {question}")
        convert_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", text_to_sql_prompt),
                ("human", "Question: {question}"),
            ]
        )
        structured_llm = llm.with_structured_output(ConvertToSQL)
        sql_generator = convert_prompt | structured_llm
        result = sql_generator.invoke({"question": question})
        state["sql_query"] = result.sql_query
        print(f"Generated SQL query: {state['sql_query']}")
        return state

    @staticmethod
    def execute_sql(state: StatePlanner):
        sql_query = state["sql_query"].strip()
        session = SessionLocal()
        print(f"Executing SQL query: {sql_query}")
        try:
            result = session.execute(text(sql_query))
            if sql_query.lower().startswith("select"):
                rows = result.fetchall()
                columns = result.keys()
                if rows:
                    header = ", ".join(columns)
                    state["query_rows"] = [dict(zip(columns, row)) for row in rows]
                    # Format the result for readability
                    data = "\n".join([
                        ", ".join([f"{col}: {row[col]}" for col in columns]) for row in state["query_rows"]
                        ])
                    formatted_result = f"{header}\n{data}"
                else:
                    state["query_rows"] = []
                    formatted_result = "No results found."
                state["query_result"] = formatted_result
                state["sql_error"] = False
                print("SQL SELECT query executed successfully.")
            else:
                session.commit()
                state["query_result"] = "The action has been successfully completed."
                state["sql_error"] = False
                print("SQL command executed successfully.")
        except Exception as e:
            state["query_result"] = f"Error executing SQL query: {str(e)}"
            state["sql_error"] = True
            print(f"Error executing SQL query: {str(e)}")
        finally:
            session.close()
        return state

    @staticmethod
    def generate_human_readable_answer(state: StatePlanner):
        sql = state["sql_query"]
        result = state["query_result"]
        query_rows = state.get("query_rows", [])
        sql_error = state.get("sql_error", False)
        print("Generating a human-readable answer.")
        system = """
        Tu es un assistant intelligent qui convertit les résultats d'une requête SQL en réponse claire et naturelle.
        """
        if sql_error:
            # Directly relay the error message
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        f"""SQL Query: {sql}
                        Result: {result} 
                        Formule un message d'erreur clair et compréhensible en une seule phrase informant l'utilisateur du problème."""
                    ),
                ]
            )
        elif sql.lower().startswith("select"):
            if not query_rows:
                # Handle cases with no orders
                generate_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        (
                            "human",
                            f"""SQL Query: {sql}
                            Result: {result} 
                            Formule une réponse claire et compréhensible à la question initiale en une seule phrase, et mentionne qu'aucune donnée n'a été trouvée."""
                            ),
                    ]
                )
            else:
                # Handle displaying orders
                generate_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        (
                            "human",
                            f"""SQL Query: {sql}
                            Result: {result}
                            Formule une réponse claire et compréhensible à la question initiale en une seule phrase en utilisant les données récupérées par la requête SQL."""),
                    ]
                )
        else:
            # Handle non-select queries
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        f"""SQL Query: {sql}
                        Result: {result}
                        Formule un message de confirmation clair et compréhensible en une seule phrase, commençant par confirmant que la demande de l'utilisateur a été traitée avec succès."""
                        ),
                ]
            )
        human_response = generate_prompt | llm | StrOutputParser()
        answer = human_response.invoke({})
        state["query_result"] = answer
        print("Generated human-readable answer.")
        print(state)
        return state
    
    @staticmethod
    def regenerate_query(state: StatePlanner):
        question = state["question"]
        print("Regenerating the SQL query by rewriting the question.")
        system = """Tu es un assistant qui reformule une question originale pour permettre des requêtes SQL plus précises. Assure-toi que tous les détails nécessaires sont préservés afin de récupérer des données complètes et précises.
        """
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"Question originale : {question}\nReformule la question pour permettre des requêtes SQL plus précises, en veillant à ce que tous les détails nécessaires soient préservés.",
                ),
            ]
        )
        llm = llm
        structured_llm = llm.with_structured_output(RewrittenQuestion)
        rewriter = rewrite_prompt | structured_llm
        rewritten = rewriter.invoke({})
        state["question"] = rewritten.question
        state["attempts"] += 1
        print(f"Rewritten question: {state['question']}")
        return state
    
    @staticmethod
    def regenerate_query(state: StatePlanner):
        question = state["question"]
        print("Regenerating the SQL query by rewriting the question.")
        system = """Tu es un assistant qui reformule une question originale pour permettre des requêtes SQL plus précises. Assure-toi que tous les détails nécessaires, tels que les jointures de tables, sont préservés afin de récupérer des données complètes et précises. Si la date et l'heure ne sont pas précisées dans une question qui concerne une période précise ajoute les. Par exemple pour "A quelle heure ai-je cours demain ?", tu dois transformer le "demain" en une date précise.
        """
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"Question originale : {question}\nReformule la question pour permettre des requêtes SQL plus précises, en veillant à ce que tous les détails nécessaires soient préservés.",
                ),
            ]
        )
        structured_llm = llm.with_structured_output(RewrittenQuestion)
        rewriter = rewrite_prompt | structured_llm
        rewritten = rewriter.invoke({})
        state["question"] = rewritten.question
        state["attempts"] += 1
        print(f"Rewritten question: {state['question']}")
        return state
    
    @staticmethod
    def generate_funny_response(state: StatePlanner):
        question = state["question"]
        print("Generating a funny response for an unrelated question.")
        system = """Tu es un assistant charmant et drôle qui répond de manière ludique à une question hors sujet. L'utilisateur est censé poser des questions relatives sur l'emploi du temps de son université."""
        funny_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", f"Question hors sujet : {question}"),
            ]
        )
        funny_response = funny_prompt | llm | StrOutputParser()
        message = funny_response.invoke({})
        state["query_result"] = message
        print("Generated funny response.")
        return state

    @staticmethod   
    def end_max_iterations(state: StatePlanner):
        state["query_result"] = "Veuillez réessayer"
        print("Maximum attempts reached. Ending the workflow.")
        return state

    @staticmethod  
    def relevance_router(state: StatePlanner):
        if state["relevance"].lower() == "relevant":
            return "convert_to_sql"
        else:
            return "generate_funny_response"
    @staticmethod     
    def check_attempts_router(state: StatePlanner):
        if state["attempts"] < 3:
            return "convert_to_sql"
        else:
            return "end_max_iterations"
        
    @staticmethod  
    def execute_sql_router(state: StatePlanner):
        if not state.get("sql_error", False):
            return "generate_human_readable_answer"
        else:
            return "regenerate_query"
    
    async def ask_SmartPlanner(self, query: str):
        return await self.graph.ainvoke({
            "question": query,
            "query_rows": [],
            "query_result": "",
            "sql_query": "",
            "current_user": "",
            "attempts": 0,
            "relevance": "",
            "sql_error": False,
            })

# -------------- TESTS
# import asyncio
# if __name__ == "__main__":
#     instance = SmartPlanner()
#     result = asyncio.run(instance.ask_SmartPlanner("Qui est Mathis Jacq ?"))
#     print(result)
# import datetime
# agent = SmartPlanner()
# exemple_state = {
#     "question": "A quelle heure j'ai cours demain ?",
#     "sql_query": "",
#     "query_result": "",
#     "query_rows": [],
#     "attempts": 0,
#     "relevance": "",
#     "sql_error": False
# }
# agent.regenerate_query(exemple_state)['question']
