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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Cargar variables desde el archivo .env
load_dotenv()

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="Tutor GPC TCE", page_icon="🩺", layout="centered")

st.title("🧠 Tutor Socrático: GPC TCE Pediátrico")
st.markdown("""
Esta herramienta evalúa el razonamiento clínico basado en la **GPC de CENETEC**. Di "Hola" para iniciar el caso clínico. Recuerda: el tutor solo guiará tu análisis, no te dará respuestas directas. ¡Buena suerte! 
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
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
        msgs = StreamlitChatMessageHistory(key="chat_history")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Adopta el rol de un Médico Adjunto de Urgencias Pediátricas, seco, estricto y de pocas palabras. "
                "Tu objetivo es evaluar si el residente (usuario) aplica correctamente la GPC de CENETEC.\n\n"
                "PROTOCOLO DE RESPUESTA (ESTRICTO):\n"
                "1. Inicio del Caso: Presentale al residente SOLO el motivo de ingreso breve. Debe ser un caso típico de TCE pediátrico. "
                "Detente ahí y pregunta: '¿Cuál es tu abordaje inicial?'.\n"
                "2. Inicio Aleatorio: Cada vez que inicies un caso, TÚ debes elegir internamente una gravedad distinta: "
                "TCE Leve (Glasgow 14-15), Moderado (9-13) o Severo (3-8). Varía el mecanismo de lesión.\n"
                "3. Prohibición de Monotema: NO te limites a convulsiones. Explora escenarios de: \n"
                "   - Intervalo lúcido (sugestivo de Epidural).\n"
                "   - Deterioro progresivo (sugestivo de Subdural).\n"
                "   - Vómito persistente o cefalea intensa en TCE leve.\n"
                "   - Datos de fractura de base de cráneo (ojos de mapache, signo de Battle).\n"
                "4. Entrega de Datos (Cero Números en Glasgow): \n"
                "   - Para Signos Vitales: Provee cifras exactas (FC, TA, FR, Temp, Saturación).\n"
                "   - Para Neurología: Tienes PROHIBIDO dar el puntaje numérico o el desglose (O, V, M). Describe la CONDUCTA "
                "     física (ej. 'El niño solo abre los ojos al pellizcarlo, emite sonidos incomprensibles y retira la mano al dolor').\n"
                "5. El Residente Calcula: Tras describir la conducta, pregunta: '¿Qué puntaje de Glasgow asignas y qué severidad tiene según la GPC?'.\n"
                "6. Trampa de Decisión (Obligatoria cada 5 turnos): Sugiere una contraindicación o un error de manejo con mucha seguridad. "
                "Si el residente acepta el error, repréndelo citando la GPC. Si lo refuta, reconoce su juicio.\n"
                "7. Detención en Seco: No te respondas a ti mismo. Una respuesta tuya = Una descripción/evaluación + Una pregunta corta.\n"
                "8. Enfoque Clínico: Prohibido hablar de logística, administración o procesos post-alta.\n\n"
                "DOCUMENTO FUENTE (GPC):\n{context}"
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "input": RunnablePassthrough(), "history": lambda x: msgs.messages}
        )

        def generar_respuesta_con_contexto(data):
            docs_crudos = data["context"]
            contexto_formateado = format_docs(docs_crudos)
            respuesta_texto = (prompt | llm | StrOutputParser()).invoke({
                "context": contexto_formateado,
                "input": data["input"],
                "history": data["history"]
            })
            return {"answer": respuesta_texto, "context": docs_crudos}

        chain = setup_and_retrieval | generar_respuesta_con_contexto

        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if user_input := st.chat_input("Escribe tu análisis clínico..."):
            st.chat_message("human").write(user_input)
            with st.spinner("Analizando la GPC..."):
                # 1. Invocamos la cadena. LangChain suele devolver un diccionario aquí.
                respuesta_completa = chain.invoke(user_input)
                # OJO AQUÍ: Dependiendo de cómo armaste tu "chain" arriba, 
                # las llaves del diccionario suelen llamarse "answer" y "context" 
                # (o a veces "result" y "source_documents").
                texto_respuesta = respuesta_completa["answer"]
                documentos_fuente = respuesta_completa["context"]
                # 2. Mostramos el mensaje del asistente
                with st.chat_message("assistant"):
                    st.write(texto_respuesta)
                    # 3. Agregamos el botón desplegable con la evidencia
                    if documentos_fuente:
                        with st.expander("🔍 Ver evidencia clínica en la GPC"):
                            for i, doc in enumerate(documentos_fuente):
                                # Intentamos sacar el número de página si PyMuPDF lo guardó
                                pagina = doc.metadata.get("page", "N/A")
                                st.markdown(f"**Fragmento extraído {i+1} (Página {pagina}):**")
                                st.info(doc.page_content)
                # 4. Guardamos en el historial SOLO el texto (no los documentos) 
                # para no saturar la memoria de la conversación
                msgs.add_user_message(user_input)
                msgs.add_ai_message(texto_respuesta)
else:
    st.info("👋 Por favor, ingresa tu API Key para comenzar.")

if st.sidebar.button("Reiniciar Caso"):
    msgs.clear()
    st.rerun()