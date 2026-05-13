# 🧠 Tutor Socrático IA: Razonamiento Clínico en TCE Pediátrico

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tutor-gpc-tce-dzmkwpxpnjlk5krqy6efye.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://langchain.com/)

## 📌 Descripción del Proyecto
Este repositorio contiene el código fuente de un sistema de tutoría inteligente (Tutor Socrático) diseñado para evaluar y guiar el razonamiento clínico de médicos en formación frente a casos de Traumatismo Craneoencefálico (TCE) Pediátrico. 

El sistema está estrictamente fundamentado en la **Guía de Práctica Clínica (GPC) de CENETEC (México)** y utiliza una arquitectura de Generación Aumentada por Recuperación (RAG) para asegurar que las respuestas del modelo de lenguaje estén ancladas a la evidencia médica oficial, minimizando alucinaciones.

## 🔬 Metodología y Framework de Evaluación
El diseño de los *prompts* y la lógica de corrección del tutor están inspirados en metodologías recientes para la evaluación de Modelos de Lenguaje Grande (LLMs) en entornos clínicos:
* **Framework de Goodell et al. (2025):** Para la identificación y corrección de errores de interpretación clínica y asignación de criterios (ej. cálculo de la Escala de Coma de Glasgow).
* **MEDCALC-BENCH:** Para la evaluación del razonamiento cuantitativo y estratificación de riesgo en medicina basada en evidencia.

## ⚙️ Arquitectura Técnica
* **Interfaz:** Streamlit
* **Orquestación:** LangChain
* **Modelo de Lenguaje (LLM):** `llama-3.3-70b-versatile` (vía Groq API) optimizado para razonamiento lógico y baja latencia.
* **Embeddings:** `intfloat/multilingual-e5-large` (HuggingFace) para una alta comprensión semántica del texto médico en español.
* **Base de Datos Vectorial:** ChromaDB
* **Procesamiento de Documentos:** PyMuPDF para la extracción de texto del documento fuente.

## 🚀 Despliegue en Vivo
La aplicación se encuentra hospedada y disponible para su uso público en Streamlit Community Cloud:
👉 **[Acceder al Tutor Socrático GPC TCE](https://tutor-gpc-tce-dzmkwpxpnjlk5krqy6efye.streamlit.app/)**

## 💻 Instalación y Uso Local (Reproducibilidad)
Para investigadores que deseen replicar este entorno localmente:

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/miguelcervera9927/tutor-gpc-tce.git](https://github.com/miguelcervera9927/tutor-gpc-tce.git)
   cd tutor-gpc-tce
   ```
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configura tus credenciales:
   Crea un archivo `.env` en la raíz del proyecto y añade tu API Key de Groq:
   ```text
   GROQ_API_KEY=tu_clave_aqui
   ```
4. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

## ⚠️ Aviso Legal
*Esta herramienta es un prototipo con fines estrictamente académicos y de investigación. No es un dispositivo médico y no debe utilizarse para tomar decisiones diagnósticas o terapéuticas reales en el entorno clínico.*
