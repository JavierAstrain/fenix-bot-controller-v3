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
import datetime as dt
from typing import Dict, Any, Optional, Tuple
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
            st.error("Secrets USER/PASSWORD no configurados. Agrega USER y PASSWORD en secrets.toml / Cloud.")
            return
        if username == try_user and password == try_pass:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Credenciales incorrectas")

if not st.session_state.authenticated:
    login()
    st.stop()

# ---------------------------
# ESTADO PERSISTENTE
# ---------------------------
if "historial" not in st.session_state:
    st.session_state.historial = []
if "data" not in st.session_state:
    st.session_state.data = None
if "sheet_url" not in st.session_state:
    st.session_state.sheet_url = ""
if "__ultima_vista__" not in st.session_state:
    st.session_state["__ultima_vista__"] = None
# Visualización
if "max_cats_grafico" not in st.session_state:
    st.session_state.max_cats_grafico = 18
if "top_n_grafico" not in st.session_state:
    st.session_state.top_n_grafico = 12
# Aliases de columnas aprendidos
if "aliases" not in st.session_state:
    st.session_state.aliases = {}

# ---------------------------
# CARGA DE DATOS (CACHE)
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
    system = f"Eres un controller financiero experto del Taller Fénix (desabolladura y pintura). Tono: {'breve, ejecutivo (5–7 frases)' if 'Ejecutivo' in modo else 'analítico, concreto (hasta 12 frases)'}."
    messages = [{"role": "system", "content": system}]
    for h in st.session_state.historial[-8:]:
        messages.append({"role": "user", "content": h["pregunta"]})
        messages.append({"role": "assistant", "content": h["respuesta"]})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content

# ---------------------------
# NORMALIZACIÓN & UTILIDADES
# ---------------------------
def _norm(s: str) -> str:
    s = str(s).replace("\u00A0", " ").strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def find_col(df: pd.DataFrame, name: str) -> Optional[str]:
    if not name:
        return None
    # alias persistidos
    alias = st.session_state.aliases.get(_norm(name))
    if alias and alias in df.columns:
        return alias
    tgt = _norm(name)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    # sugerencias simples si no hay match exacto
    candidates = [c for c in df.columns if _norm(name) in _norm(c) or _norm(c).startswith(tgt[:4])]
    if candidates:
        return candidates[0]
    return None

def _find_col(df, name: str) -> Optional[str]:  # alias para planner
    return find_col(df, name)

def _build_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    schema = {}
    for hoja, df in data.items():
        if df is None or df.empty:
            continue
        cols = []
        samples = {}
        for c in df.columns:
            cols.append(str(c))
            vals = df[c].dropna().astype(str).head(3).tolist()
            if vals:
                samples[str(c)] = vals
        schema[hoja] = {"columns": cols, "samples": samples}
    return schema

# ---------------------------
# FILTROS DE NEGOCIO (slicers)
# ---------------------------
def deduce_cols(data: Dict[str, pd.DataFrame]) -> Tuple[Optional[str], Optional[str]]:
    fecha_col = None
    cliente_col = None
    for d in data.values():
        for c in d.columns:
            c2 = _norm(c)
            if not fecha_col and ("fecha" in c2 or "mes" in c2):
                fecha_col = c
            if not cliente_col and ("cliente" in c2):
                cliente_col = c
    return fecha_col, cliente_col

def aplicar_filtros(df: pd.DataFrame, fecha_col: Optional[str], cliente_col: Optional[str],
                    rango: Optional[Tuple[dt.date, dt.date]], cliente_txt: str) -> pd.DataFrame:
    out = df.copy()
    if fecha_col and fecha_col in out.columns and rango:
        out[fecha_col] = pd.to_datetime(out[fecha_col], errors="coerce")
        if isinstance(rango, tuple) and len(rango) == 2:
            ini = pd.to_datetime(rango[0])
            fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1)
            out = out[(out[fecha_col] >= ini) & (out[fecha_col] < fin)]
    if cliente_txt and cliente_col and cliente_col in out.columns:
        out = out[out[cliente_col].astype(str).str.contains(cliente_txt, case=False, na=False)]
    return out

# ---------------------------
# SELECCIÓN DE VISUALIZACIÓN
# ---------------------------
def _choose_chart_auto(df: pd.DataFrame, cat_col: str, val_col: str) -> str:
    """
    'torta' | 'barras' | 'linea' | 'table'
    - Temporal -> línea
    - IDs (patente/folio/doc/factura/OC/orden/boleta/cotización) -> tabla
    - ≤6 categorías -> torta
    - ≤ max_cats_grafico -> barras
    - resto -> tabla
    """
    cat_norm = _norm(cat_col)
    if pd.api.types.is_datetime64_any_dtype(df[cat_col]) or "fecha" in cat_norm or "mes" in cat_norm:
        return "linea"
    id_hints = ["patente", "folio", "nro", "numero", "número", "doc", "documento",
                "factura", "boleta", "oc", "orden", "presupuesto", "cotizacion", "cotización"]
    if any(h in cat_norm for h in id_hints):
        return "table"
    nunique = df[cat_col].nunique(dropna=False)
    if nunique <= 6:
        return "torta"
    if nunique <= st.session_state.get("max_cats_grafico", 18):
        return "barras"
    return "table"

# ---------------------------
# FORMATO MONETARIO (CLP)
# ---------------------------
def _fmt_pesos(x, pos=None):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"${int(round(x)):,}".replace(",", ".")
    except Exception:
        return str(x)

# ---------------------------
# VISUALIZACIONES
# ---------------------------
def _export_fig(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf.read()

def mostrar_grafico_torta(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals)
                 .groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))
    fig, ax = plt.subplots()
    ax.pie(resumen.values, labels=[str(x) for x in resumen.index], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    st.pyplot(fig)
    st.download_button("⬇️ Descargar gráfico (PNG)", _export_fig(fig), "grafico.png", "image/png")

def _barras_vertical(resumen, col_categoria, col_valor, titulo):
    fig, ax = plt.subplots()
    bars = ax.bar(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    ax.set_ylabel(col_valor)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    ax.tick_params(axis='x', rotation=45)
    for lbl in ax.get_xticklabels(): lbl.set_ha('right')
    for b in bars:
        y = b.get_height()
        if np.isfinite(y):
            ax.annotate(_fmt_pesos(y), xy=(b.get_x()+b.get_width()/2, y),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button("⬇️ Descargar gráfico (PNG)", _export_fig(fig), "grafico.png", "image/png")

def _barras_horizontal(resumen, col_categoria, col_valor, titulo):
    fig, ax = plt.subplots()
    bars = ax.barh(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    ax.set_xlabel(col_valor)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    for b in bars:
        x = b.get_width()
        if np.isfinite(x):
            ax.annotate(_fmt_pesos(x), xy=(x, b.get_y()+b.get_height()/2),
                        xytext=(5, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button("⬇️ Descargar gráfico (PNG)", _export_fig(fig), "grafico.png", "image/png")

def mostrar_grafico_barras(df, col_categoria, col_valor, titulo=None, top_n=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals)
                 .groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))

    # Outlier -> tabla
    try:
        if len(resumen) >= 8:
            top_val = float(resumen.iloc[0])
            med_val = float(np.median(resumen.values))
            if med_val > 0 and top_val / med_val >= 3.0:
                st.info("Distribución muy desbalanceada: mostrando tabla en lugar de barras.")
                mostrar_tabla(df, col_categoria, col_valor, titulo)
                return
    except Exception:
        pass

    if top_n is None:
        top_n = st.session_state.get("top_n_grafico", 12)
    recorte = False
    if len(resumen) > top_n:
        resumen = resumen.head(top_n)
        recorte = True

    labels = resumen.index.astype(str)
    if labels.str.len().mean() > 10:
        _barras_horizontal(resumen, col_categoria, col_valor, titulo)
    else:
        _barras_vertical(resumen, col_categoria, col_valor, titulo)
    if recorte:
        st.caption(f"Mostrando Top‑{top_n}. Usa una tabla para el detalle completo.")

def mostrar_tabla(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals)
                 .groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False).reset_index())
    resumen.columns = [str(col_categoria).title(), str(col_valor).title()]
    col_val = resumen.columns[1]
    resumen[col_val] = resumen[col_val].apply(_fmt_pesos)
    total_val = vals.sum(skipna=True)
    total_row = pd.DataFrame({resumen.columns[0]: ["TOTAL"], col_val: [_fmt_pesos(total_val)]})
    resumen = pd.concat([resumen, total_row], ignore_index=True)
    st.markdown(f"### 📊 {titulo if titulo else f'{col_val} por {col_categoria}'}")
    st.dataframe(resumen, use_container_width=True)
    st.download_button("⬇️ Descargar tabla (CSV)", resumen.to_csv(index=False).encode("utf-8"),
                       "tabla.csv", "text/csv")

# ---------------------------
# PARSER (texto → render)
# ---------------------------
def parse_and_render_instructions(respuesta_texto: str, data_dict: dict, filtros):
    patt = re.compile(r'(grafico_torta|grafico_barras|tabla)(?:@([^\s:]+))?\s*:\s*([^\n\r]+)', re.IGNORECASE)

    def safe_plot(plot_fn, hoja, df, cat_raw, val_raw, titulo):
        cat = find_col(df, cat_raw); val = find_col(df, val_raw)
        if not cat or not val:
            st.warning(f"❗ No se pudo generar en '{hoja}'. Revisar columnas: '{cat_raw}' y '{val_raw}'.")
            return
        try:
            fecha_col, cliente_col, rango, cliente = filtros
            df_fil = aplicar_filtros(df, fecha_col, cliente_col, rango, cliente)
            plot_fn(df_fil, cat, val, titulo)
            st.session_state.aliases[_norm(cat_raw)] = cat
            st.session_state.aliases[_norm(val_raw)] = val
        except Exception as e:
            st.error(f"Error generando visualización en '{hoja}': {e}")

    for m in patt.finditer(respuesta_texto):
        kind = m.group(1).lower()
        hoja_sel = m.group(2)
        body = m.group(3).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]

        if kind in ("grafico_torta", "grafico_barras"):
            if len(parts) != 3: 
                st.warning("Instrucción de gráfico inválida."); continue
            cat_raw, val_raw, title = parts
            if hoja_sel and hoja_sel in data_dict:
                safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                          hoja_sel, data_dict[hoja_sel], cat_raw, val_raw, title)
            else:
                ok=False
                for hoja, df in data_dict.items():
                    if find_col(df, cat_raw) and find_col(df, val_raw):
                        safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                                  hoja, df, cat_raw, val_raw, title)
                        ok=True
                if not ok: st.warning("No se pudo generar el gráfico (verifica columnas).")
        else:  # tabla
            if len(parts) not in (2,3): st.warning("Instrucción de tabla inválida."); continue
            cat_raw, val_raw = parts[0], parts[1]; title = parts[2] if len(parts)==3 else None
            def draw(df, hoja):
                cat = find_col(df, cat_raw); val = find_col(df, val_raw)
                if cat and val:
                    fecha_col, cliente_col, rango, cliente = filtros
                    df_fil = aplicar_filtros(df, fecha_col, cliente_col, rango, cliente)
                    mostrar_tabla(df_fil, cat, val, titulo=title or f"Tabla: {val} por {cat} ({hoja})")
                    st.session_state.aliases[_norm(cat_raw)] = cat
                    st.session_state.aliases[_norm(val_raw)] = val
                    return True
                return False
            if hoja_sel and hoja_sel in data_dict:
                if not draw(data_dict[hoja_sel], hoja_sel):
                    st.warning(f"No se pudo generar la tabla en '{hoja_sel}'.")
            else:
                ok=False
                for hoja, df in data_dict.items(): ok = draw(df, hoja) or ok
                if not ok: st.warning("No se pudo generar la tabla (verifica columnas).")

# ---------------------------
# PLANNER (IA → JSON)
# ---------------------------
def plan_from_llm(pregunta: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Eres un controller financiero. Te doy el ESQUEMA de las hojas (columnas y ejemplos) y una PREGUNTA del usuario.
Devuélveme SOLO un JSON (sin explicaciones) con la mejor acción para responder, con esta forma:

{{
  "action": "table" | "chart" | "text",
  "sheet": "<nombre_hoja_o_vacia_si_no_aplica>",
  "category_col": "<col cat o vacio>",
  "value_col": "<col valor o vacio>",
  "date_col": "<col fecha si aplica o vacio>",
  "agg": "sum" | "avg" | "count",
  "chart": "barras" | "torta" | "linea" | "auto",
  "title": "<titulo sugerido>"
}}

Reglas:
- Usa NOMBRES EXACTOS del esquema (insensible a mayúsculas).
- Si piden “por …”, úsalo como categoría.
- Si la categoría es un identificador (patente, folio, nº documento, factura, orden, OC, etc.), prefiere "action":"table".
- Si no se indica tipo de gráfico, usa "chart":"auto".
- Si dudas del valor, usa monto/importe/neto/total.
- Si es puramente textual, usa "action":"text".
- Usa respuestas cortas y accionables.

ESQUEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

PREGUNTA:
{pregunta}
"""
    raw = ask_gpt(prompt).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m: return {}
    try:
        plan = json.loads(m.group(0))
        return plan if isinstance(plan, dict) else {}
    except Exception:
        return {}

# ---------------------------
# EJECUTOR DEL PLAN
# ---------------------------
def execute_plan(plan: Dict[str, Any], data: Dict[str, Any], filtros) -> bool:
    action = plan.get("action")
    if action not in ("table", "chart", "text"):
        return False
    if action == "text":
        return False

    sheet = plan.get("sheet") or ""
    cat = plan.get("category_col") or ""
    val = plan.get("value_col") or ""
    date_col = plan.get("date_col") or ""
    chart = (plan.get("chart") or "auto").lower()
    title = plan.get("title") or None

    hojas = [sheet] if sheet in data else list(data.keys())
    for h in hojas:
        df = data[h]
        if df is None or df.empty: continue

        cat_real  = _find_col(df, cat)  if cat  else None
        val_real  = _find_col(df, val)  if val  else None
        date_real = _find_col(df, date_col) if date_col else None

        if not val_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["monto","importe","neto","total","facturacion","ingreso","venta"]):
                    val_real = c; break
        if not cat_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["fecha","mes","patente","cliente","tipo","estado","proceso","servicio","vehiculo","unidad"]):
                    cat_real = c; break
        if not val_real or not cat_real: continue

        # aplicar filtros
        fecha_col, cliente_col, rango, cliente = filtros
        df_fil = aplicar_filtros(df, fecha_col, cliente_col, rango, cliente)

        if action == "chart" and chart == "auto":
            chart = _choose_chart_auto(df_fil, cat_real, val_real)

        if action == "table" or chart == "table":
            mostrar_tabla(df_fil, cat_real, val_real, title)
            st.session_state["__ultima_vista__"] = {"sheet": h, "cat": cat_real, "val": val_real, "type": "tabla"}
            return True

        if action == "chart":
            if chart == "barras":
                mostrar_grafico_barras(df_fil, cat_real, val_real, title)
            elif chart == "torta":
                mostrar_grafico_torta(df_fil, cat_real, val_real, title)
            elif chart == "linea":
                df2 = df_fil.copy()
                if not pd.api.types.is_datetime64_any_dtype(df2[cat_real]):
                    try: df2[cat_real] = pd.to_datetime(df2[cat_real], errors="coerce")
                    except Exception: pass
                if pd.api.types.is_datetime64_any_dtype(df2[cat_real]):
                    serie = (df2.set_index(cat_real).groupby(pd.Grouper(freq="M"))[val_real].sum().dropna())
                    fig, ax = plt.subplots()
                    ax.plot(serie.index, serie.values, marker="o")
                    ax.set_title(title or f"{val_real} por tiempo")
                    ax.set_ylabel(val_real)
                    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
                    fig.autofmt_xdate()
                    st.pyplot(fig)
                    st.download_button("⬇️ Descargar gráfico (PNG)", _export_fig(fig), "grafico.png", "image/png")
                else:
                    mostrar_tabla(df_fil, cat_real, val_real, title or f"Tabla: {val_real} por {cat_real}")
            else:
                mostrar_grafico_barras(df_fil, cat_real, val_real, title)

            st.session_state["__ultima_vista__"] = {"sheet": h, "cat": cat_real, "val": val_real, "type": f"chart:{chart}"}
            return True
    return False

# ---------------------------
# UI
# ---------------------------
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 📁 Datos")
    fuente = st.radio("Fuente", ["Excel", "Google Sheets"], key="k_fuente")

    if fuente == "Excel":
        file = st.file_uploader("Sube un Excel", type=["xlsx", "xls"], key="k_excel")
        if file: st.session_state.data = load_excel(file)
    else:
        with st.form(key="form_gsheet"):
            url = st.text_input("URL de Google Sheet", value=st.session_state.sheet_url, key="k_url")
            conectar = st.form_submit_button("Conectar")
        if conectar and url:
            try:
                nuevo = load_gsheet(st.secrets["GOOGLE_CREDENTIALS"], url)
                if nuevo and len(nuevo) > 0:
                    st.session_state.sheet_url = url
                    st.session_state.data = nuevo
                    st.success("Google Sheet conectado.")
                else:
                    st.warning("La hoja no tiene datos.")
            except Exception as e:
                st.error(f"Error conectando Google Sheet: {e}")

    st.markdown("### ⚙️ Preferencias")
    st.session_state.modo_respuesta = st.radio("Modo de respuesta", ["Ejecutivo (breve)", "Analítico (detallado)"], horizontal=False)
    st.session_state.max_cats_grafico = st.number_input("Máx. categorías para graficar", min_value=6, max_value=200, value=st.session_state.max_cats_grafico, step=1)
    st.session_state.top_n_grafico = st.number_input("Top‑N por defecto (barras)", min_value=5, max_value=100, value=st.session_state.top_n_grafico, step=1)

data = st.session_state.data

with col2:
    if data:
        # Slicers de negocio
        fecha_col, cliente_col = deduce_cols(data)
        st.markdown("### 🎛️ Filtros")
        rango = st.date_input("Rango de fechas",
                              value=(dt.date.today().replace(day=1), dt.date.today())) if fecha_col else None
        cliente_txt = st.text_input("Cliente contiene…", value="") if cliente_col else ""

        tabs = st.tabs(["📈 KPIs", "📄 Vista previa", "🤖 Consulta IA", "🧠 Historial"])
        # KPIs
        with tabs[0]:
            try:
                kpis = analizar_datos_taller(data, (fecha_col, cliente_col, rango, cliente_txt))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ingresos ($)", f"{int(round(kpis['ingresos'])):,}".replace(",", "."))
                c2.metric("Costos ($)",   f"{int(round(kpis['costos'])):,}".replace(",", "."))
                c3.metric("Margen ($)",   f"{int(round(kpis['margen'])):,}".replace(",", "."))
                c4.metric("Margen %",     f"{kpis['margen_pct']:.1f}%")
            except Exception as e:
                st.info("Carga un archivo para ver KPIs.")

        # Vista previa
        with tabs[1]:
            st.markdown("### 📄 Hojas")
            for name, df in data.items():
                dfv = aplicar_filtros(df, fecha_col, cliente_col, rango, cliente_txt)
                st.markdown(f"#### 📘 {name}  •  filas: {len(dfv)}")
                st.dataframe(dfv.head(10))

        # Consulta IA
        with tabs[2]:
            st.markdown("### 🤖 Consulta")
            pregunta = st.text_area("Pregunta")
            cta1, cta2 = st.columns(2)
            if cta1.button("📊 Análisis General Automático"):
                try:
                    analisis = analizar_datos_taller(data, (fecha_col, cliente_col, rango, cliente_txt))
                except Exception as e:
                    st.warning(f"⚠️ Problema al calcular análisis: {e}")
                    analisis = {}
                prompt = f"""
Eres un controller financiero senior.
Con base en los datos calculados (reales) a continuación, entrega un análisis profesional, directo y accionable.
Incluye UNA instrucción de visualización si ayuda (en una línea):
- grafico_torta:col_categoria|col_valor|titulo
- grafico_barras:col_categoria|col_valor|titulo
- tabla:col_categoria|col_valor
No inventes datos.

Datos calculados:
{json.dumps(analisis, ensure_ascii=False, indent=2)}
"""
                respuesta = ask_gpt(prompt)
                st.markdown(respuesta)
                st.session_state.historial.append({"pregunta": "Análisis general", "respuesta": respuesta})
                parse_and_render_instructions(respuesta, data, (fecha_col, cliente_col, rango, cliente_txt))

            if cta2.button("Responder") and pregunta:
                schema = _build_schema(data)
                plan = plan_from_llm(pregunta, schema)
                executed = False
                if plan:
                    executed = execute_plan(plan, data, (fecha_col, cliente_col, rango, cliente_txt))
                if not executed:
                    prompt = f"""
Responde como controller financiero. Si puedes, incluye UNA instrucción exacta:
- grafico_torta:cat|val|titulo
- grafico_barras:cat|val|titulo
- tabla:cat|val[|titulo]
Para categorías tipo ID (patente/folio/doc/factura/oc), prefiere tabla. No expliques el método si puedes dar el resultado.
Pregunta: {pregunta}
Esquema: {json.dumps(schema, ensure_ascii=False)}
"""
                    respuesta = ask_gpt(prompt)
                    st.markdown(respuesta)
                    st.session_state.historial.append({"pregunta": pregunta, "respuesta": respuesta})
                    parse_and_render_instructions(respuesta, data, (fecha_col, cliente_col, rango, cliente_txt))

        # Historial
        with tabs[3]:
            if st.session_state.historial:
                for i, h in enumerate(st.session_state.historial[-12:], 1):
                    st.markdown(f"**Q{i}:** {h['pregunta']}")
                    st.markdown(f"**A{i}:** {h['respuesta']}")
            else:
                st.info("Aún no hay historial en esta sesión.")
