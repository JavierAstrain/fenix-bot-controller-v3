import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from openai import OpenAI
from analizador import analizar_datos_taller, generar_tabla, generar_grafico

st.set_page_config(page_title="Fénix Bot Controller", layout="wide")

# ------------------------
# IA: función central
# ------------------------
def ask_gpt(prompt: str) -> str:
    """
    Cliente robusto OpenAI (SDK v1.x):
    - Lee la API key desde st.secrets / env
    - Soporta opcionalmente OPENAI_BASE_URL / OPENAI_ORG
    - Errores explicativos (no rompe la app)
    """
    api_key = (
        st.secrets.get("OPENAI_API_KEY")
        or st.secrets.get("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        st.error("⚠️ Falta `OPENAI_API_KEY` en Secrets. Ve a Settings → Secrets y agrégalo.")
        return "⚠️ Configura `OPENAI_API_KEY` para habilitar la IA."

    api_key = str(api_key).strip().strip('"').strip("'")

    org = st.secrets.get("OPENAI_ORG") or os.getenv("OPENAI_ORG")
    base_url = st.secrets.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    kwargs = {"api_key": api_key}
    if org:
        kwargs["organization"] = str(org).strip()
    if base_url:
        kwargs["base_url"] = str(base_url).strip()

    try:
        client = OpenAI(**kwargs)
    except Exception as e:
        st.error(f"⚠️ No se pudo inicializar OpenAI: {e}")
        return "⚠️ Error inicializando OpenAI."

    modo = st.session_state.get("modo_respuesta", "Ejecutivo (breve)")
    system = (
        "Eres un controller financiero experto. Tono: "
        + ("breve, ejecutivo (5–7 frases)" if "Ejecutivo" in modo else "analítico, concreto")
    )

    messages = [{"role": "system", "content": system}]
    for h in st.session_state.historial[-8:]:
        messages += [{"role": "user", "content": h["pregunta"]},
                     {"role": "assistant", "content": h["respuesta"]}]
    messages.append({"role": "user", "content": prompt})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"⚠️ Fallo en la petición a OpenAI: {e}")
        return "⚠️ No pude completar la consulta a la IA."

# ------------------------
# Estado
# ------------------------
if "historial" not in st.session_state:
    st.session_state.historial = []
if "data" not in st.session_state:
    st.session_state.data = None

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("⚙️ Configuración")
st.sidebar.radio("Modo de respuesta:", ["Ejecutivo (breve)", "Analítico"], key="modo_respuesta")

with st.sidebar.expander("🔎 Diagnóstico OpenAI"):
    if st.button("Probar conexión OpenAI"):
        try:
            api_key = (
                st.secrets.get("OPENAI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            client = OpenAI(api_key=str(api_key).strip())
            models = client.models.list()
            st.success(f"OK. Modelos disponibles: {len(models.data)}")
        except Exception as e:
            st.error(f"No se pudo conectar: {e}")

# ------------------------
# Cargar datos
# ------------------------
st.title("🤖 Fénix Bot Controller IA")

archivo = st.file_uploader("📂 Sube un Excel/CSV", type=["xlsx", "xls", "csv"])
if archivo:
    try:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo, sheet_name=None)  # todas las hojas
        st.session_state.data = df
        st.success("✅ Datos cargados correctamente.")
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")

# ------------------------
# KPIs + Dashboard
# ------------------------
if st.session_state.data is not None:
    st.subheader("📊 KPIs Generales")

    if isinstance(st.session_state.data, dict):
        df_all = pd.concat(st.session_state.data.values(), ignore_index=True)
    else:
        df_all = st.session_state.data

    # Si hay columna de ingresos
    kpi_cols = [c for c in df_all.columns if "monto" in c.lower() or "ingreso" in c.lower()]
    if kpi_cols:
        col = kpi_cols[0]
        total = df_all[col].sum()
        st.metric("Ingresos Totales", f"${total:,.0f}")
        # gráfico simple
        fig, ax = plt.subplots()
        df_all[col].plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Distribución de {col}")
        st.pyplot(fig)
    else:
        st.info("No se detectaron columnas de ingresos/montos para mostrar KPIs.")

# ------------------------
# Consulta IA
# ------------------------
st.subheader("💬 Consulta IA")
pregunta = st.text_input("Escribe tu consulta")
if st.button("Responder") and pregunta:
    contenido = ""
    if isinstance(st.session_state.data, dict):
        for name, df in st.session_state.data.items():
            contenido += f"Hoja: {name}\n{df.head(50).to_string(index=False)}\n\n"
    else:
        contenido = st.session_state.data.head(50).to_string(index=False)

    prompt = f"""
Datos disponibles:\n{contenido}\n
Pregunta del usuario: {pregunta}
"""
    r = ask_gpt(prompt)
    st.markdown(r)
    st.session_state.historial.append({"pregunta": pregunta, "respuesta": r})

# ------------------------
# Historial
# ------------------------
with st.expander("🧠 Historial de la sesión"):
    for h in st.session_state.historial:
        st.write(f"**Tú:** {h['pregunta']}")
        st.write(f"**IA:** {h['respuesta']}")
