import json
import re
import html
import pandas as pd
from langchain_community.utilities import SQLDatabase
import os
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy import create_engine
from datetime import datetime

with open(r'Agent\SmartPlanner\data\edt.json', "r", encoding="utf-8") as f:
    edt = json.load(f)

def clean_event(i, event):
    # reformate la description pour récupérer la salle et le cours
    description = html.unescape(event['description'])
    lines = list(filter(str.strip, re.split(r'<br\s*/?>|\n|\r', description)))
    match = re.search(r'-\s*(.*?)\s*\[', lines[2])
    if match:
        nom_cours=match.group(1)
    else: nom_cours = ""

    formations = []
    for line in lines:
        text = html.unescape(line)
        if "Ingénierie Statistique, Actuariat et Data Science ISADS" in text:
            formations.append("M2 Saclay Ingénierie Statistique, Actuariat et Data Science ISADS")
        if "Mathématiques et Apprentissage Statistique" in text:
            formations.append("M2 Saclay Mathématiques et Apprentissage Statistique" + " Math&AS")
            
    return {
        "id": str(i),
        "type": event.get("eventCategory", ""),  # CM/TD, etc.
        "debut": event["start"],
        "fin": event["end"],
        "cours": nom_cours,
        "salle": re.sub(r'\s+([.?!])', r'\1', lines[1].split("-")[0]),
        "batiment": "".join(s for s in event.get("sites") or []),
        "formation" : "; ".join(formations),
        "module": ", ".join(event.get("modules") or [])
    }
    
cleaned_events = [clean_event(i, e) for i, e in enumerate(edt)]
cleaned_events = sorted(cleaned_events, key=lambda x: datetime.fromisoformat(x["debut"]))

DB_PATH = r"Agent\SmartPlanner\data\edt.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
db = SQLDatabase(engine=engine)

Base = declarative_base()
class Edt(Base):
    __tablename__ = "edt"
    id = Column(String, primary_key=True)  # Clé primaire obligatoire
    type=Column(String, index=True)
    debut=Column(String, index=True)
    fin=Column(String, index=True)
    cours=Column(String, index=True)
    salle=Column(String, index=True)
    batiment=Column(String, index=True)
    formation=Column(String, index=True)
    module=Column(String, index=True)

def init_db():
    # Création de la table (si elle n'existe pas)
    Base.metadata.create_all(engine)

    # Suppose que cleaned_events est ta liste de dictionnaires
    with Session(engine) as session:
        objets = [Edt(**e) for e in cleaned_events]
        session.add_all(objets)
        session.commit()

if __name__ == "__main__":
    if not os.path.exists(r"Agent.SmartPlanner.data\.edt.db"):
        init_db()
    else:
        print("edt.db déjà présent dans le path")


# ---------- APERCU DES DONNEES ----------
edt_df = pd.DataFrame(cleaned_events)   # exporte au format dataframe
edt_df["formation"].unique()


edt_df.debut.value_counts()

test = edt_df[(edt_df["debut"]>="2025-03-01") & (edt_df["fin"]<="2025-03-31")]
test.batiment.value_counts()
