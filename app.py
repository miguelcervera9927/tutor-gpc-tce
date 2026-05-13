import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Cargar variables desde el archivo .env
load_dotenv()

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="Tutor GPC TCE", page_icon="🩺", layout="centered")

st.title("🧠 Tutor Socrático: GPC TCE Pediátrico")
st.markdown("""
Esta herramienta evalúa el razonamiento clínico basado en la **GPC de CENETEC**. 
""")

# --- PROCESAMIENTO DEL CONOCIMIENTO (RAG) ---
@st.cache_resource
def configurar_cerebro():
    # Nombre de archivo actualizado según tu solicitud
    nombre_archivo = "gpc_tce_pediatrico.pdf"
    
    if not os.path.exists(nombre_archivo):
        st.error(f"Error: No se encontró el archivo '{nombre_archivo}' en la carpeta.")
        return None
    
    loader = PyMuPDFLoader(nombre_archivo)
    documentos = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = splitter.split_documents(documentos)
    
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# --- LÓGICA DEL TUTOR ---
# Intenta obtener la clave desde el archivo .env o la barra lateral
# Intenta obtener la clave de Streamlit Secrets (Nube) o de .env (Local)
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Ingresa tu Groq API Key (o configúrala en .env)", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    retriever = configurar_cerebro()
    
    if retriever:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        msgs = StreamlitChatMessageHistory(key="chat_history")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Eres un tutor experto en neurocirugía pediátrica. Tu misión es guiar al estudiante "
                "en la resolución de casos de TCE basándote ÚNICAMENTE en la GPC proporcionada.\n\n"
                "METODOLOGÍA:\n"
                "1. Plantea el caso por etapas. No des la solución.\n"
                "2. Evalúa la respuesta del estudiante. Si hay errores de interpretación o asignación, "
                "corrígelos citando la GPC.\n"
                "3. Mantén un tono profesional.\n\n"
                "Contexto de la GPC:\n{context}"
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough(), "history": lambda x: msgs.messages}
            | prompt
            | llm
            | StrOutputParser()
        )

        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if user_input := st.chat_input("Escribe tu análisis clínico..."):
            st.chat_message("human").write(user_input)
            
            with st.spinner("Analizando la GPC..."):
                response = chain.invoke(user_input)
                st.chat_message("assistant").write(response)
                msgs.add_user_message(user_input)
                msgs.add_ai_message(response)
else:
    st.info("👋 Por favor, ingresa tu API Key para comenzar.")

if st.sidebar.button("Reiniciar Caso"):
    msgs.clear()
    st.rerun()