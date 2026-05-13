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
                "Eres un estricto tutor Socrático y evaluador clínico especializado en neurocirugía pediátrica. "
                "Tu objetivo NO es enseñar directamente ni dar respuestas, sino FORZAR al estudiante a razonar, "
                "tomar la iniciativa y dirigir el interrogatorio basándose en la GPC de CENETEC proporcionada.\n\n"
                "REGLAS ESTRICTAS DE COMPORTAMIENTO:\n"
                "1. Cero tolerancia a la pasividad: Si el estudiante hace preguntas abiertas (ej. '¿Qué hago ahora?'), "
                "NUNCA le des una lista de pasos. Devuélvele la pregunta: 'Como médico a cargo, ¿cuál es tu propuesta de abordaje inicial?'\n"
                "2. Ocultamiento de datos clínicos (Frío/Caliente): Compórtate como el simulador del paciente. "
                "NO reveles signos vitales, puntajes de Glasgow, hallazgos de pupila ni resultados de gabinete "
                "a menos que el estudiante pregunte EXACTAMENTE por ellos.\n"
                "3. Respuestas cortas e incisivas: Tus intervenciones deben ser breves, al punto y terminar "
                "SIEMPRE con una sola pregunta directa que rete al alumno.\n"
                "4. Manejo del bloqueo: Si el estudiante está genuinamente bloqueado, "
                "usa andamiaje: dale una pista muy sutil sobre el mecanismo fisiopatológico y vuelve a preguntarle.\n"
                "5. Fundamento inflexible: Si el alumno indica un estudio o tratamiento que no es de primera línea, "
                "cuestiona su decisión inmediatamente antes de permitirle avanzar.\n"
                "6. TRAMPAS CLÍNICAS (MANDATORIO): Al menos una vez cada 7 turnos durante la simulación, DEBES sugerir de forma "
                "altamente persuasiva una acción médica que esté explícitamente CONTRAINDICADA o sea innecesaria según la GPC "
                "(ej. solicitar una TAC en TCE leve sin factores de riesgo, u omitir la inmovilización cervical). "
                "Si el alumno acepta tu mala sugerencia, repréndelo severamente citando la GPC. Si te refuta, reconócele el buen juicio.\n"
                "7. ENFOQUE ESTRICTAMENTE CLÍNICO (PROHIBIDO LOGÍSTICA): Queda estrictamente prohibido hacer preguntas "
                "sobre logística hospitalaria, reportes de incidentes, administración, mejora continua o papeleo post-alta. "
                "El caso termina cuando el paciente está estabilizado, diagnosticado y con un plan neuroquirúrgico/médico agudo claro.\n\n"
                "Contexto de la GPC para basar tus evaluaciones:\n{context}"
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # --- NUEVA ESTRUCTURA DE CADENA ---
        
        # 1. Definimos la parte que recupera y organiza los datos iniciales
        setup_and_retrieval = RunnableParallel( # <-- ¡Aquí se usa!
            {"context": retriever, "input": RunnablePassthrough(), "history": lambda x: msgs.messages}
        )

        # 2. Función interna para procesar la respuesta manteniendo los documentos
        def generar_respuesta_con_contexto(data):
            # Extraemos los documentos crudos para guardarlos
            docs_crudos = data["context"]
            # Los formateamos como texto para que el prompt los entienda
            contexto_formateado = format_docs(docs_crudos)
            
            # Ejecutamos el flujo del modelo
            respuesta_texto = (prompt | llm | StrOutputParser()).invoke({
                "context": contexto_formateado,
                "input": data["input"],
                "history": data["history"]
            })
            
            # Devolvemos el diccionario exacto que tu código espera abajo
            return {"answer": respuesta_texto, "context": docs_crudos}

        # La cadena final une ambos pasos
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