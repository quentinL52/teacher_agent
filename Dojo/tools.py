import requests
import json
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_groq import ChatGroq  
from youtube_search  import YoutubeSearch

load_dotenv()
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
FAISS_INDEX_PATH = "faiss_course_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def Youtube(query: str) -> str:
    results = YoutubeSearch(query).to_json()
    data = json.loads(results)
    url_list = [f"https://www.youtube.com{video['url_suffix']}" for video in data['videos']]       
    return "\n".join(url_list)


def _internet_search(query: str) -> str:
    """
    Effectue une recherche sur Internet et retourne une chaîne de caractères formatée 
    avec les titres, les liens et les descriptions des résultats.
    """
    if not SERPER_API_KEY:
        return "Erreur: SERPER_API_KEY non configurée"

    search = SerpAPIWrapper(serpapi_api_key=SERPER_API_KEY, search_engine="google")
    
    try:
        results_dict = search.results(query)
        
        if not results_dict or "organic_results" not in results_dict or not results_dict["organic_results"]:
            return "Aucun résultat trouvé."
            
        formatted_results = []
        for result in results_dict["organic_results"][:5]:
            title = result.get("title", "Pas de titre")
            link = result.get("link", "#")
            snippet = result.get("snippet", "Pas de description.")
            formatted_results.append(f"- Titre: {title}\n  Lien: {link}\n  Description: {snippet}\n")
            
        return "\n".join(formatted_results) if formatted_results else "Aucun résultat trouvé."

    except Exception as e:
        return f"Une erreur est survenue lors de la recherche : {e}"

@tool
def generate_course(topic: str) -> str:
    """
    Utilise cet outil pour générer un cours détaillé et structuré sur un sujet (topic) spécifique
    quand un utilisateur demande explicitement un cours, un topo, un résumé complet, etc.
    L'outil s'occupe de tout le processus, de la recherche à la rédaction finale du cours.
    Les cours doivent intégrer des liens vers des sites pertinents ou des vidéos youtube.
    """
    print(f"--- Action: Génération d'un cours complet sur '{topic}' ---")
    
    # Perform searches
    search_results = _internet_search(topic)
    video_search_results = Youtube(topic)
    resource_query = f"meilleurs tutoriels vidéos youtube et ressources en ligne pour {topic}"
    resource_search_results = _internet_search(resource_query)
    
    # Create a more effective prompt
    prompt = f"""
    En tant qu'enseignant expert, rédige un cours détaillé et structuré sur le sujet suivant : '{topic}'.

    Le cours doit être clair, bien organisé, et facile à comprendre pour un débutant. Structure-le avec :
    1.  Une **introduction** qui présente le sujet et son importance.
    2.  Plusieurs **sections thématiques** détaillées avec des explications claires et des exemples concrets.
    3.  Une **conclusion** qui résume les points clés.

    Pour construire ce cours, inspire-toi des informations générales ci-dessous :
    --- INFORMATIONS DE BASE ---
    {search_results}

    Tout au long du cours, tu dois **enrichir le contenu en intégrant de manière fluide et pertinente** les liens vers les ressources web et les vidéos listées ci-dessous. Ne te contente pas de lister les liens à la fin, mais insère-les là où ils sont les plus utiles pour illustrer un point ou approfondir une section. Pour chaque lien, indique clairement à quoi il mène.

    --- RESSOURCES WEB ET VIDÉOS PERTINENTES ---
    Ressources Web: {resource_search_results}
    Vidéos YouTube: {video_search_results}

    Rédige le cours final.
    """
    
    final_course = llm.invoke(prompt).content
    return final_course

@tool
def answer_from_saved_courses(question: str) -> str:
    """utilise cet outils pour repondre au question de l'utilisateur sur les cours enregistré sur le vector store FAISS
    """
    print(f"--- Action: Recherche dans les cours enregistrés pour la question '{question}' ---")
    if not os.path.exists(FAISS_INDEX_PATH):
        return "google est ton ami"

    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retrieved_docs = vector_store.similarity_search(question, k=4)
    
    if not retrieved_docs:
        return "google est ton ami"
        
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""En te basant STRICTEMENT sur le contexte suivant, réponds à la question de l'utilisateur.
    Contexte: {context}
    Question: {question}
    """
    
    final_answer = llm.invoke(prompt).content
    return final_answer

from langchain_core.documents import Document

@tool
def save_course(topic: str, content: str) -> str:
    """
    Utilise cet outil pour enregistrer un cours détaillé et structuré sur un sujet (topic) spécifique
    quand un utilisateur demande explicitement d'enregistrer un cours, un topo, un résumé complet, etc.
    L'outil s'occupe de tout le processus, de l'embedding a l'indexation sur la vector'.
    """
    print(f"--- Action: Enregistrement du cours sur le sujet : '{topic}' ---")    
    if not content:
        return "Erreur : aucun contenu fourni pour l'enregistrement."
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=content)])

    if os.path.exists(FAISS_INDEX_PATH):
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(FAISS_INDEX_PATH)
    return f"Le cours sur '{topic}' a bien été enregistré."




def get_tools_for_agent():
    return [generate_course, answer_from_saved_courses, save_course]
