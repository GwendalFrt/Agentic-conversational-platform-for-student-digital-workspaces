from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import llm

class RouterAgent:
    def __init__(self):
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un classificateur d'intention. Tu dois classer la question dans l'une de ces catégories : 
             - 'cours' si elle concerne une notion académique comme une définition, démonstration, formule, etc.
             - 'emploi_du_temps' si elle concerne une organisation de planning, horaires, ou cours programmés
             - 'UVSQ' si elle concerne des informations sur l'Université de Versailles Saint-Quentin en Yvelines (UVSQ) qui peuvent être disponibles 
             sur son site internet
             - 'autre' si elle ne concerne aucune des 3 thématiques précédentes.
             Réponds uniquement par 'cours' ou 'emploi_du_temps' ou 'UVSQ' ou 'autre'."""),
            ("human", "{question}")
        ])

        self.intent_classifier = intent_prompt | llm | StrOutputParser()

    async def ask_router(self, question: str) -> str:
        return self.intent_classifier.invoke({"question": question})
