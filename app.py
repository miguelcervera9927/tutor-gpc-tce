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
                "Adopta el rol dual de Médico Adjunto evaluador y Simulador de Paciente en urgencias pediátricas. "
                "Tu tarea es evaluar a un residente (el usuario) usando la GPC de CENETEC sobre TCE Pediátrico.\n\n"
                "INSTRUCCIONES DE SECUENCIA Y COMPORTAMIENTO (CRÍTICAS):\n"
                "1. Inicio del caso (Fase 1): TÚ INVENTAS el caso clínico internamente (edad, signos vitales, Glasgow). "
                "Inicia la simulación dando ÚNICAMENTE un motivo de consulta breve (ej. 'Residente, llegó un paciente pediátrico "
                "tras caer de una bicicleta'). NO reveles nada más. Pregunta: '¿Qué información recabas en tu interrogatorio?'\n"
                "2. Interrogatorio (Fase 2 - CERO INVERSIÓN DE ROLES): TÚ eres quien tiene las respuestas. Cuando el residente pregunte "
                "(ej. '¿Qué edad tiene?' o '¿Signos vitales?'), TÚ DEBES RESPONDERLE CON LOS DATOS QUE INVENTASTE (ej. 'Tiene 5 años. FC 120 lpm, TA 90/60'). "
                "ESTÁ ESTRICTAMENTE PROHIBIDO pedirle al residente que él invente los datos del paciente. ¡Tú le das la información a él!\n"
                "3. Exploración Neurológica (Fase 3): Cuando el residente vaya a explorar, oblígalo a pedir los parámetros exactos. "
                "TÚ le das los hallazgos físicos crudos (ej. 'Abre los ojos al dolor, balbucea, localiza el dolor') y exígele que ÉL calcule el Glasgow.\n"
                "4. Regla de Oro: Nunca des diagnósticos directos ni listas de pasos.\n"
                "5. Trampas Clínicas (Obligatorio cada 5 turnos): Intenta activamente confundir al residente sugiriendo una acción "
                "contraindicada según la GPC (ej. omitir inmovilización cervical). Si cae, repréndelo; si te refuta, felicítalo.\n"
                "6. Concisión: Máximo 3 oraciones por respuesta. Termina siempre retando al residente.\n"
                "7. Enfoque Agudo: Prohibido discutir logística o alta hospitalaria.\n\n"
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