import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gspread
import io
import json
import re
import unicodedata
from typing import Dict, Any, Optional
from google.oauth2.service_account import Credentials
from openai import OpenAI
from analizador import analizar_datos_taller

st.set_page_config(layout="wide", page_title="Controller Financiero IA")

# ---------------------------
# LOGIN
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.markdown("## 🔐 Iniciar sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        try_user = st.secrets.get("USER", None)
        try_pass = st.secrets.get("PASSWORD", None)
        if try_user is None or try_pass is None:
            st.error("Secrets USER/PASSWORD no configurados.")
            return
        if username == try_user and password == try_pass:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Credenciales incorrectas")

if not st.session_state.authenticated:
    login(); st.stop()

# ---------------------------
# ESTADO PERSISTENTE
# ---------------------------
if "historial" not in st.session_state: st.session_state.historial = []
if "data" not in st.session_state:       st.session_state.data = None
if "sheet_url" not in st.session_state:  st.session_state.sheet_url = ""
if "__ultima_vista__" not in st.session_state: st.session_state["__ultima_vista__"] = None
if "max_cats_grafico" not in st.session_state: st.session_state.max_cats_grafico = 18
if "top_n_grafico" not in st.session_state:    st.session_state.top_n_grafico = 12
if "aliases" not in st.session_state:          st.session_state.aliases = {}
if "modo_respuesta" not in st.session_state:   st.session_state.modo_respuesta = "Ejecutivo (breve)"

# ---------------------------
# CARGA (CACHE)
# ---------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_excel(file):
    return pd.read_excel(file, sheet_name=None)

@st.cache_data(show_spinner=False, ttl=300)
def load_gsheet(json_keyfile: str, sheet_url: str):
    creds_dict = json.loads(json_keyfile)
    scope = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    return {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sheet.worksheets()}

# ---------------------------
# OPENAI
# ---------------------------
def ask_gpt(prompt: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    modo = st.session_state.get("modo_respuesta", "Ejecutivo (breve)")
    system = f"Eres un controller financiero experto del Taller Fénix. Tono: {'breve, ejecutivo (5–7 frases)' if 'Ejecutivo' in modo else 'analítico, concreto (hasta 12 frases)'}."
    messages = [{"role": "system", "content": system}]
    for h in st.session_state.historial[-8:]:
        messages.append({"role": "user", "content": h["pregunta"]})
        messages.append({"role": "assistant", "content": h["respuesta"]})
    messages.append({"role": "user", "content": prompt})
    res = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.2)
    return res.choices[0].message.content

# ---------------------------
# UTILIDADES (normalización, formato)
# ---------------------------
def _norm(s: str) -> str:
    s = str(s).replace("\u00A0"," ").strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+',' ', s)
    return s.lower()

def find_col(df: pd.DataFrame, name: str):
    if not name: return None
    alias = st.session_state.aliases.get(_norm(name))
    if alias and alias in df.columns: return alias
    tgt = _norm(name)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    candidates = [c for c in df.columns if tgt in _norm(c) or _norm(c).startswith(tgt[:4])]
    return candidates[0] if candidates else None

def _fmt_pesos(x, pos=None):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return ""
        return f"${int(round(x)):,}".replace(",", ".")
    except Exception:
        return str(x)

def _export_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf.read()

# ---------------------------
# VISUALIZACIONES
# ---------------------------
def mostrar_tabla(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False).reset_index())
    resumen.columns = [str(col_categoria).title(), str(col_valor).title()]
    col_val = resumen.columns[1]
    resumen[col_val] = resumen[col_val].apply(_fmt_pesos)
    total_val = vals.sum(skipna=True)
    resumen.loc[len(resumen)] = ["TOTAL", _fmt_pesos(total_val)]
    st.markdown(f"### 📊 {titulo if titulo else f'{col_val} por {col_categoria}'}")
    st.dataframe(resumen, use_container_width=True)
    st.download_button("⬇️ Descargar tabla (CSV)", resumen.to_csv(index=False).encode("utf-8"),
                       "tabla.csv", "text/csv")

def mostrar_grafico_torta(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))
    fig, ax = plt.subplots()
    ax.pie(resumen.values, labels=[str(x) for x in resumen.index], autopct='%1.1f%%', startangle=90)
    ax.axis('equal'); ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    st.pyplot(fig)
    st.download_button("⬇️ Descargar gráfico (PNG)", _export_fig(fig), "grafico.png", "image/png")

def _barras_vertical(resumen, col_categoria, col_valor, titulo):
    fig, ax = plt.subplots()
    bars = ax.bar(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    ax.set_ylabel(col_valor); ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    ax.tick_params(axis='x', rotation=45)
    for lbl in ax.get_xticklabels(): lbl.set_ha('right')
    for b in bars:
        y = b.get_height()
        if np.isfinite(y):
            ax.annotate(_fmt_pesos(y), (b.get_x()+b.get_width()/2, y), textcoords="offset points",
                        xytext=(0,3), ha='center', va='bottom', fontsize=8)
    fig.tight_layout(); st.pyplot(fig)
    st.download_button("⬇️ Descargar gráfico (PNG)", _export_fig(fig), "grafico.png", "image/png")

def _barras_horizontal(resumen, col_categoria, col_valor, titulo):
    fig, ax = plt.subplots()
    bars = ax.barh(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    ax.set_xlabel(col_valor); ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    for b in bars:
        x = b.get_width()
        if np.isfinite(x):
            ax.annotate(_fmt_pesos(x), (x, b.get_y()+b.get_height()/2), textcoords="offset points",
                        xytext=(5,0), ha='left', va='center', fontsize=8)
    fig.tight_layout(); st.pyplot(fig)
    st.download_button("⬇️ Descargar gráfico (PNG)", _export_fig(fig), "grafico.png", "image/png")

def mostrar_grafico_barras(df, col_categoria, col_valor, titulo=None, top_n=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))
    # outlier -> tabla
    try:
        if len(resumen) >= 8:
            top_val = float(resumen.iloc[0]); med = float(np.median(resumen.values))
            if med > 0 and top_val / med >= 3.0:
                st.info("Distribución desbalanceada: muestro tabla para mejor lectura.")
                mostrar_tabla(df, col_categoria, col_valor, titulo); return
    except Exception:
        pass
    if top_n is None: top_n = st.session_state.get("top_n_grafico", 12)
    recorte = False
    if len(resumen) > top_n:
        resumen = resumen.head(top_n); recorte = True
    labels = resumen.index.astype(str)
    if labels.str.len().mean() > 10: _barras_horizontal(resumen, col_categoria, col_valor, titulo)
    else:                            _barras_vertical(resumen, col_categoria, col_valor, titulo)
    if recorte: st.caption(f"Mostrando Top‑{top_n}. Usa tabla para el detalle completo.")

# ---------------------------
# AUTO-CHOOSER
# ---------------------------
def _choose_chart_auto(df: pd.DataFrame, cat_col: str, val_col: str) -> str:
    cat_norm = _norm(cat_col)
    # IDs -> tabla
    id_hints = ["patente","folio","nro","numero","número","doc","documento","factura","boleta","oc","orden","presupuesto","cotizacion","cotización"]
    if any(h in cat_norm for h in id_hints): return "table"
    nunique = df[cat_col].nunique(dropna=False)
    if nunique <= 6: return "torta"
    if nunique <= st.session_state.get("max_cats_grafico", 18): return "barras"
    return "table"

# ---------------------------
# PARSER (texto → render)
# ---------------------------
def parse_and_render_instructions(respuesta_texto: str, data_dict: dict, cliente_txt: str):
    patt = re.compile(r'(grafico_torta|grafico_barras|tabla)(?:@([^\s:]+))?\s*:\s*([^\n\r]+)', re.IGNORECASE)

    def apply_client(df):
        if not cliente_txt: return df
        cliente_col = next((c for c in df.columns if "cliente" in _norm(c)), None)
        if not cliente_col: return df
        out = df.copy()
        return out[out[cliente_col].astype(str).str.contains(cliente_txt, case=False, na=False)]

    def safe_plot(plot_fn, hoja, df, cat_raw, val_raw, titulo):
        cat = find_col(df, cat_raw); val = find_col(df, val_raw)
        if not cat or not val:
            st.warning(f"❗ No se pudo generar en '{hoja}'. Revisar columnas: '{cat_raw}' y '{val_raw}'."); return
        try:
            plot_fn(apply_client(df), cat, val, titulo)
            st.session_state.aliases[_norm(cat_raw)] = cat
            st.session_state.aliases[_norm(val_raw)] = val
        except Exception as e:
            st.error(f"Error generando visualización en '{hoja}': {e}")

    for m in patt.finditer(respuesta_texto):
        kind = m.group(1).lower()
        hoja_sel = m.group(2)
        body = m.group(3).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]
        if kind in ("grafico_torta","grafico_barras"):
            if len(parts) != 3: st.warning("Instrucción de gráfico inválida."); continue
            cat_raw, val_raw, title = parts
            if hoja_sel and hoja_sel in data_dict:
                safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                          hoja_sel, data_dict[hoja_sel], cat_raw, val_raw, title)
            else:
                ok=False
                for hoja, df in data_dict.items():
                    if find_col(df, cat_raw) and find_col(df, val_raw):
                        safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                                  hoja, df, cat_raw, val_raw, title); ok=True
                if not ok: st.warning("No se pudo generar el gráfico (verifica columnas).")
        else:
            if len(parts) not in (2,3): st.warning("Instrucción de tabla inválida."); continue
            cat_raw, val_raw = parts[0], parts[1]; title = parts[2] if len(parts)==3 else None
            def draw(df, hoja):
                cat = find_col(df, cat_raw); val = find_col(df, val_raw)
                if cat and val:
                    mostrar_tabla(apply_client(df), cat, val, titulo or f"Tabla: {val} por {cat} ({hoja})")
                    st.session_state.aliases[_norm(cat_raw)] = cat
                    st.session_state.aliases[_norm(val_raw)] = val
                    return True
                return False
            if hoja_sel and hoja_sel in data_dict:
                if not draw(data_dict[hoja_sel], hoja_sel): st.warning(f"No se pudo generar la tabla en '{hoja_sel}'.")
            else:
                ok=False
                for hoja, df in data_dict.items(): ok = draw(df, hoja) or ok
                if not ok: st.warning("No se pudo generar la tabla (verifica columnas).")

# ---------------------------
# PLANNER (LLM)
# ---------------------------
def _build_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    schema = {}
    for hoja, df in data.items():
        if df is None or df.empty: continue
        cols = []; samples = {}
        for c in df.columns:
            cols.append(str(c))
            vals = df[c].dropna().astype(str).head(3).tolist()
            if vals: samples[str(c)] = vals
        schema[hoja] = {"columns": cols, "samples": samples}
    return schema

def plan_from_llm(pregunta: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Eres un controller financiero. Devuelve SOLO un JSON con el mejor plan:
{{
  "action": "table" | "chart" | "text",
  "sheet": "<nombre hoja o vacío>",
  "category_col": "<cat>",
  "value_col": "<val>",
  "date_col": "",
  "agg": "sum",
  "chart": "barras" | "torta" | "linea" | "auto",
  "title": "<título>"
}}
Reglas:
- Usa nombres EXACTOS del esquema (insensible a mayúsculas).
- Si la categoría es un identificador (patente/folio/doc/factura/oc), prefiere "action":"table".
- Si no se indica tipo de gráfico, usa "chart":"auto".
ESQUEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}
PREGUNTA:
{pregunta}
"""
    raw = ask_gpt(prompt).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m: return {}
    try:
        plan = json.loads(m.group(0)); return plan if isinstance(plan, dict) else {}
    except Exception: return {}

def execute_plan(plan: Dict[str, Any], data: Dict[str, Any], cliente_txt: str) -> bool:
    action = plan.get("action")
    if action not in ("table","chart","text"): return False
    if action == "text": return False
    sheet = plan.get("sheet") or ""
    cat = plan.get("category_col") or ""
    val = plan.get("value_col") or ""
    chart = (plan.get("chart") or "auto").lower()
    title = plan.get("title") or None

    def apply_client(df):
        if not cliente_txt: return df
        cliente_col = next((c for c in df.columns if "cliente" in _norm(c)), None)
        if not cliente_col: return df
        out = df.copy()
        return out[out[cliente_col].astype(str).str.contains(cliente_txt, case=False, na=False)]

    hojas = [sheet] if sheet in data else list(data.keys())
    for h in hojas:
        df = data[h]
        if df is None or df.empty: continue
        cat_real = find_col(df, cat) if cat else None
        val_real = find_col(df, val) if val else None
        if not val_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["monto","importe","neto","total","facturacion","ingreso","venta"]):
                    val_real = c; break
        if not cat_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["patente","cliente","tipo","estado","proceso","servicio","vehiculo","unidad","mes","fecha"]):
                    cat_real = c; break
        if not (cat_real and val_real): continue

        df_fil = apply_client(df)

        if action == "chart" and chart == "auto":
            chart = _choose_chart_auto(df_fil, cat_real, val_real)

        if action == "table" or chart == "table":
            mostrar_tabla(df_fil, cat_real, val_real, title)
            st.session_state["__ultima_vista__"] = {"sheet": h, "cat": cat_real, "val": val_real, "type":"tabla"}
            return True

        if chart == "barras":   mostrar_grafico_barras(df_fil, cat_real, val_real, title)
        elif chart == "torta":  mostrar_grafico_torta(df_fil, cat_real, val_real, title)
        elif chart == "linea":  mostrar_grafico_barras(df_fil, cat_real, val_real, title)  # fallback seguro
        else:                   mostrar_grafico_barras(df_fil, cat_real, val_real, title)

        st.session_state["__ultima_vista__"] = {"sheet": h, "cat": cat_real, "val": val_real, "type":f"chart:{chart}"}
        return True
    return False

# ---------------------------
# DASHBOARD KPIs (nuevo)
# ---------------------------
def _first_col_with(df: pd.DataFrame, keys):
    for c in df.columns:
        if any(k in _norm(str(c)) for k in keys):
            return c
    return None

def _apply_client_filter_df(df: pd.DataFrame, client_txt: str) -> pd.DataFrame:
    if not client_txt:
        return df
    cli = _first_col_with(df, ["cliente"])
    if not cli:
        return df
    out = df.copy()
    return out[out[cli].astype(str).str.contains(client_txt, case=False, na=False)]

def _ing_col(df: pd.DataFrame):
    for k in ["monto", "neto", "total", "importe", "facturacion", "ingreso", "venta", "principal"]:
        c = _first_col_with(df, [k])
        if c is not None:
            return c
    return None

def _date_col(df: pd.DataFrame):
    return _first_col_with(df, ["fecha", "mes", "emision", "emisión"])

def _count_col(df: pd.DataFrame):
    return _first_col_with(df, ["estado", "resultado", "situacion", "situación", "status"])

def render_kpi_dashboard(data: dict, cliente_txt: str):
    st.markdown("#### 📊 Dashboard")
    c1, c2, c3 = st.columns(3)

    # 1) Top-10 Clientes
    plotted1 = False
    for hoja, df in data.items():
        cli = _first_col_with(df, ["cliente"])
        val = _ing_col(df)
        if cli and val:
            df2 = _apply_client_filter_df(df, cliente_txt).copy()
            vals = pd.to_numeric(df2[val], errors="coerce")
            top = (df2.assign(__v=vals).groupby(cli, dropna=False)["__v"]
                     .sum().sort_values(ascending=False).head(10))
            with c1:
                st.caption(f"Top‑10 Clientes ({hoja})")
                _barras_horizontal(top, cli, val, titulo=None)
            plotted1 = True
            break
    if not plotted1:
        with c1: st.info("No se encontró columna de cliente para Top‑10.")

    # 2) Distribución por Estado
    plotted2 = False
    for hoja, df in data.items():
        est = _count_col(df)
        if est:
            df2 = _apply_client_filter_df(df, cliente_txt)
            dist = df2[est].astype(str).str.strip().str.title().value_counts().head(8)
            if len(dist) >= 1:
                with c2:
                    st.caption(f"Distribución por Estado ({hoja})")
                    fig, ax = plt.subplots()
                    ax.pie(dist.values, labels=dist.index.astype(str), autopct="%1.0f%%", startangle=90)
                    ax.axis("equal")
                    st.pyplot(fig)
                    st.download_button("⬇️ PNG", _export_fig(fig), "kpi_estado.png", "image/png")
                plotted2 = True
                break
    if not plotted2:
        with c2: st.info("No se encontró columna de estado para distribución.")

    # 3) Tendencia mensual de ingresos (suma multi-hoja)
    ser_total = None
    for hoja, df in data.items():
        val = _ing_col(df); fec = _date_col(df)
        if val and fec:
            df2 = _apply_client_filter_df(df, cliente_txt).copy()
            df2[fec] = pd.to_datetime(df2[fec], errors="coerce")
            df2["__v"] = pd.to_numeric(df2[val], errors="coerce")
            g = (df2.dropna(subset=[fec])
                    .set_index(fec)
                    .groupby(pd.Grouper(freq="M"))["__v"].sum())
            ser_total = g if ser_total is None else ser_total.add(g, fill_value=0)
    if ser_total is not None and len(ser_total.dropna()) >= 2:
        with c3:
            st.caption("Tendencia Mensual de Ingresos (todas las hojas con fecha)")
            fig, ax = plt.subplots()
            ax.plot(ser_total.index, ser_total.values, marker="o")
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
            fig.autofmt_xdate()
            st.pyplot(fig)
            st.download_button("⬇️ PNG", _export_fig(fig), "kpi_tendencia.png", "image/png")
    else:
        with c3: st.info("No se detectaron fechas para tendencia mensual.")

# ---------------------------
# UI
# ---------------------------
col1, col2 = st.columns([1,3])

with col1:
    st.markdown("### 📁 Datos")
    fuente = st.radio("Fuente", ["Excel","Google Sheets"], key="k_fuente")
    if fuente == "Excel":
        file = st.file_uploader("Sube un Excel", type=["xlsx","xls"], key="k_excel")
        if file: st.session_state.data = load_excel(file)
    else:
        with st.form(key="form_gsheet"):
            url = st.text_input("URL de Google Sheet", value=st.session_state.sheet_url, key="k_url")
            conectar = st.form_submit_button("Conectar")
        if conectar and url:
            try:
                st.session_state.data = load_gsheet(st.secrets["GOOGLE_CREDENTIALS"], url)
                st.session_state.sheet_url = url
                st.success("Google Sheet conectado.")
            except Exception as e:
                st.error(f"Error conectando Google Sheet: {e}")

    st.markdown("### ⚙️ Preferencias")
    st.session_state.modo_respuesta = st.radio("Modo de respuesta", ["Ejecutivo (breve)","Analítico (detallado)"])
    st.session_state.max_cats_grafico = st.number_input("Máx. categorías para graficar", 6, 200, st.session_state.max_cats_grafico)
    st.session_state.top_n_grafico = st.number_input("Top‑N por defecto (barras)", 5, 100, st.session_state.top_n_grafico)

data = st.session_state.data

with col2:
    if data:
        # Filtro opcional por cliente (genérico)
        st.markdown("### 🎛️ Filtros")
        cliente_txt = st.text_input("Cliente contiene…", value="")

        tabs = st.tabs(["📈 KPIs", "📄 Vista previa", "🤖 Consulta IA", "🧠 Historial"])

        with tabs[0]:
            kpis = analizar_datos_taller(data, cliente_txt)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ingresos ($)", f"{int(round(kpis['ingresos'])):,}".replace(",", "."))
            c2.metric("Costos ($)",   f"{int(round(kpis['costos'])):,}".replace(",", "."))
            c3.metric("Margen ($)",   f"{int(round(kpis['margen'])):,}".replace(",", "."))
            c4.metric("Margen %",     f"{(kpis['margen_pct'] or 0):.1f}%")

            c5, c6, c7 = st.columns(3)
            c5.metric("Servicios",    f"{kpis.get('servicios',0)}")
            tp = kpis.get("ticket_promedio")
            c6.metric("Ticket promedio", f"${int(round(tp)):,}".replace(",", ".") if tp else "—")
            conv = kpis.get("conversion_pct")
            c7.metric("Conversión",   f"{conv:.1f}%" if conv is not None else "—")
            lt = kpis.get("lead_time_mediano_dias")
            if lt is not None: st.caption(f"⏱️ Lead time mediano: {lt:.1f} días")

            # Mini-dashboard
            render_kpi_dashboard(data, cliente_txt)

        with tabs[1]:
            st.markdown("### 📄 Hojas")
            for name, df in data.items():
                dfv = df.copy()
                if cliente_txt:
                    cli_col = next((c for c in df.columns if "cliente" in _norm(c)), None)
                    if cli_col:
                        dfv = dfv[dfv[cli_col].astype(str).str.contains(cliente_txt, case=False, na=False)]
                st.markdown(f"#### 📘 {name} • filas: {len(dfv)}")
                st.dataframe(dfv.head(10))

        with tabs[2]:
            st.markdown("### 🤖 Consulta")
            pregunta = st.text_area("Pregunta")
            c1b, c2b = st.columns(2)
            if c1b.button("📊 Análisis General Automático"):
                analisis = analizar_datos_taller(data, cliente_txt)
                prompt = f"""
Eres un controller financiero senior.
Con base en los datos calculados (reales) a continuación, entrega un análisis profesional y accionable.
Incluye UNA instrucción si ayuda:
- grafico_torta:cat|val|titulo
- grafico_barras:cat|val|titulo
- tabla:cat|val
No inventes datos.

Datos calculados:
{json.dumps(analisis, ensure_ascii=False, indent=2)}
"""
                r = ask_gpt(prompt); st.markdown(r)
                st.session_state.historial.append({"pregunta":"Análisis general","respuesta":r})
                parse_and_render_instructions(r, data, cliente_txt)

            if c2b.button("Responder") and pregunta:
                schema = _build_schema(data)
                plan = plan_from_llm(pregunta, schema)
                executed = False
                if plan: executed = execute_plan(plan, data, cliente_txt)
                if not executed:
                    prompt = f"""
Responde como controller financiero. Si puedes, incluye UNA instrucción:
- grafico_torta:cat|val|titulo
- grafico_barras:cat|val|titulo
- tabla:cat|val[|titulo]
Para categorías tipo ID (patente/folio/doc/factura/oc), prefiere tabla.
Pregunta: {pregunta}
Esquema: {json.dumps(schema, ensure_ascii=False)}
"""
                    r = ask_gpt(prompt); st.markdown(r)
                    st.session_state.historial.append({"pregunta":pregunta,"respuesta":r})
                    parse_and_render_instructions(r, data, cliente_txt)

        with tabs[3]:
            if st.session_state.historial:
                for i, h in enumerate(st.session_state.historial[-12:], 1):
                    st.markdown(f"**Q{i}:** {h['pregunta']}")
                    st.markdown(f"**A{i}:** {h['respuesta']}")
            else:
                st.info("Aún no hay historial en esta sesión.")

