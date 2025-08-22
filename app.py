import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import json, re, unicodedata, os
from typing import Dict, Any, Optional
from analizador import analizar_datos_taller

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Controller Financiero IA", layout="wide")

# --- FORMATOS ---
def fmt_pesos(x):
    try:
        return "$" + "{:,.0f}".format(x).replace(",", ".")
    except:
        return x

def fmt_num(x):
    try:
        return "{:,.0f}".format(x).replace(",", ".")
    except:
        return x

# --- OPENAI CLIENT ---
def ask_gpt(prompt: str) -> str:
    from openai import OpenAI
    api_key = (
        st.secrets.get("OPENAI_API_KEY")
        or st.secrets.get("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        st.error("⚠️ Falta `OPENAI_API_KEY` en Secrets.")
        return "⚠️ Configura `OPENAI_API_KEY` en Secrets."

    org = st.secrets.get("OPENAI_ORG") or os.getenv("OPENAI_ORG")
    base_url = st.secrets.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if org: kwargs["organization"] = org
    if base_url: kwargs["base_url"] = base_url

    try:
        client = OpenAI(**kwargs)
    except Exception as e:
        st.error(f"No se pudo inicializar OpenAI: {e}")
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
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )
        return res.choices[0].message.content
    except Exception as e:
        st.error(f"Fallo en petición a OpenAI: {e}")
        return "⚠️ No pude completar la consulta a la IA."


# --- HELPERS ---
def _norm(s: str) -> str:
    s = str(s).replace("\u00A0", " ").strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def mostrar_tabla(df, cat_col, val_col, title=None):
    tabla = df.groupby(cat_col)[val_col].sum().reset_index()
    tabla[val_col] = tabla[val_col].apply(fmt_pesos)
    total = df[val_col].sum()
    st.markdown(f"### {title or (val_col + ' por ' + cat_col)}")
    st.dataframe(tabla, use_container_width=True)
    st.markdown(f"**Total {val_col}: {fmt_pesos(total)}**")

def mostrar_grafico_barras(df, cat_col, val_col, title=None):
    agg = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,4))
    agg.plot(kind="bar", ax=ax)
    ax.set_title(title or f"{val_col} por {cat_col}")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: fmt_pesos(x)))
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

def mostrar_grafico_torta(df, cat_col, val_col, title=None):
    agg = df.groupby(cat_col)[val_col].sum()
    fig, ax = plt.subplots()
    agg.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title(title or f"{val_col} por {cat_col}")
    st.pyplot(fig)

# --- UI ---
st.sidebar.title("⚙️ Configuración")
modo = st.sidebar.radio("Modo de respuesta IA", ["Ejecutivo (breve)", "Analítico (detallado)"])
st.session_state["modo_respuesta"] = modo

st.title("🤖 Controller Financiero IA")

if "historial" not in st.session_state:
    st.session_state.historial = []

# --- FILE UPLOAD ---
st.subheader("Carga de datos")
file = st.file_uploader("Sube archivo Excel/CSV", type=["xlsx","xls","csv"])
data = {}
if file:
    if file.name.endswith("csv"):
        data["Hoja1"] = pd.read_csv(file)
    else:
        xls = pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            data[sheet] = pd.read_excel(file, sheet_name=sheet)
    st.session_state.data = data

if "data" in st.session_state:
    st.success("✅ Datos cargados")

    # --- DASHBOARD KPIs ---
    st.header("📊 Dashboard General")
    analisis = analizar_datos_taller(st.session_state.data)

    total_ingresos = 0
    total_costos = 0
    for hoja, df in st.session_state.data.items():
        for c in df.columns:
            if "ingreso" in c.lower() or "neto" in c.lower() or "monto" in c.lower():
                total_ingresos += df[c].sum()
            if "costo" in c.lower():
                total_costos += df[c].sum()
    margen = total_ingresos - total_costos
    margen_pct = (margen / total_ingresos * 100) if total_ingresos else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Ingresos ($)", fmt_pesos(total_ingresos))
    kpi2.metric("Costos ($)", fmt_pesos(total_costos))
    kpi3.metric("Margen ($)", fmt_pesos(margen))
    kpi4.metric("Margen %", f"{margen_pct:.1f}%")

    # --- Vista previa gráfica rápida ---
    st.subheader("Vistas rápidas")
    for hoja, df in st.session_state.data.items():
        if df is not None and not df.empty and df.shape[1] >= 2:
            cols = list(df.columns)
            val_col = None
            for c in cols:
                if any(k in c.lower() for k in ["monto","neto","total","importe","ingreso"]):
                    val_col = c; break
            cat_col = cols[0]
            if val_col:
                st.write(f"**{hoja}: {val_col} por {cat_col}**")
                mostrar_grafico_barras(df, cat_col, val_col)

    # --- Consulta IA ---
    st.subheader("Consulta IA")
    pregunta = st.text_input("Haz una pregunta:")
    if st.button("Responder") and pregunta:
        schema = {"columnas": {h:list(df.columns) for h,df in st.session_state.data.items()}}
        respuesta = ask_gpt(f"Pregunta: {pregunta}\nEsquema: {schema}")
        st.markdown(respuesta)
        st.session_state.historial.append({"pregunta": pregunta, "respuesta": respuesta})

    # --- Historial ---
    st.subheader("📜 Historial")
    for h in st.session_state.historial:
        st.markdown(f"**Tú:** {h['pregunta']}")
        st.markdown(f"**Bot:** {h['respuesta']}")
