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
import os
from typing import Dict, Any, Optional
from google.oauth2.service_account import Credentials
from openai import OpenAI
from analizador import analizar_datos_taller

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide", page_title="Controller Financiero IA")

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
if "modo_respuesta" not in st.session_state:   st.session_state.modo_respuesta = "Analítico (detallado)"
if "menu_sel" not in st.session_state:         st.session_state.menu_sel = "KPIs"

# ---------------------------
# LOADERS (CACHE)
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
# OPENAI (ROBUSTO)
# ---------------------------
def _get_openai_client():
    api_key = (
        st.secrets.get("OPENAI_API_KEY")
        or st.secrets.get("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        st.error("⚠️ Falta `OPENAI_API_KEY` en Secrets.")
        return None
    api_key = str(api_key).strip().strip('"').strip("'")
    org = st.secrets.get("OPENAI_ORG") or os.getenv("OPENAI_ORG")
    base_url = st.secrets.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if org: kwargs["organization"] = str(org).strip()
    if base_url: kwargs["base_url"] = str(base_url).strip()
    try:
        return OpenAI(**kwargs)
    except Exception as e:
        st.error(f"No se pudo inicializar OpenAI: {e}")
        return None

def ask_gpt(messages) -> str:
    client = _get_openai_client()
    if client is None: return "⚠️ Error inicializando OpenAI."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Fallo en la petición a OpenAI: {e}")
        return "⚠️ No pude completar la consulta a la IA."

# --- Diagnóstico detallado de OpenAI (para la sección Diagnóstico IA) ---
def diagnosticar_openai():
    """Devuelve un dict con el estado de la API, prueba de chat y señales de cuota."""
    res = {
        "api_key_present": False,
        "organization_set": False,
        "base_url_set": False,
        "list_models_ok": False,
        "chat_ok": False,
        "quota_ok": None,           # True/False/None
        "usage_tokens": None,
        "error": None
    }
    # 1) Presencia de credenciales
    key = (
        st.secrets.get("OPENAI_API_KEY")
        or st.secrets.get("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    res["api_key_present"] = bool(key)
    res["organization_set"] = bool(st.secrets.get("OPENAI_ORG") or os.getenv("OPENAI_ORG"))
    res["base_url_set"] = bool(st.secrets.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"))

    if not res["api_key_present"]:
        res["error"] = "No se encontró OPENAI_API_KEY."
        return res

    # 2) Cliente + listar modelos
    client = _get_openai_client()
    if client is None:
        res["error"] = "No se pudo inicializar el cliente OpenAI."
        return res

    try:
        _ = client.models.list()
        res["list_models_ok"] = True
    except Exception as e:
        res["error"] = f"Error listando modelos: {e}"
        # A veces aquí ya aparece insufficient_quota o invalid_api_key
        msg = str(e).lower()
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            res["quota_ok"] = False
        return res

    # 3) Mini-prueba de chat
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Ping de diagnóstico. Responde: OK."}],
            temperature=0,
            max_tokens=5
        )
        res["chat_ok"] = True
        # `usage` puede no estar siempre
        try:
            if hasattr(r, "usage") and r.usage and r.usage.total_tokens is not None:
                res["usage_tokens"] = int(r.usage.total_tokens)
        except Exception:
            pass
        # Si llegó aquí, no hubo error de cuota
        res["quota_ok"] = True if res["quota_ok"] is None else res["quota_ok"]
    except Exception as e:
        msg = str(e)
        res["error"] = f"Error en prueba de chat: {msg}"
        low = msg.lower()
        if "insufficient_quota" in low or "exceeded your current quota" in low:
            res["quota_ok"] = False
        else:
            res["quota_ok"] = None
    return res

# ---------------------------
# DIAGNÓSTICO CONEXIÓN OPENAI (SIDEBAR)
# ---------------------------
with st.sidebar.expander("🔎 Diagnóstico OpenAI"):
    if st.button("Probar conexión OpenAI"):
        client = _get_openai_client()
        if client:
            try:
                models = client.models.list()
                st.success(f"OK. Modelos disponibles: {len(models.data)}")
            except Exception as e:
                st.error(f"No se pudo listar modelos: {e}")

# ---------------------------
# UTILIDADES
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
    buf.seek(0); return buf.read()

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

def _barras_vertical(resumen, col_categoria, col_valor, titulo):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
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
    fig.tight_layout(); st.pyplot(fig); st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

def _barras_horizontal(resumen, col_categoria, col_valor, titulo):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    fig, ax = plt.subplots()
    bars = ax.barh(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    ax.set_xlabel(col_valor); ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    for b in bars:
        x = b.get_width()
        if np.isfinite(x):
            ax.annotate(_fmt_pesos(x), (x, b.get_y()+b.get_height()/2), textcoords="offset points",
                        xytext=(5,0), ha='left', va='center', fontsize=8)
    fig.tight_layout(); st.pyplot(fig); st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

def mostrar_grafico_barras(df, col_categoria, col_valor, titulo=None, top_n=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))
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
    if recorte: st.caption(f"Mostrando Top-{top_n}. Usa tabla para el detalle completo.")

def mostrar_grafico_torta(df, col_categoria, col_valor, titulo=None):
    import matplotlib.pyplot as plt
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))
    fig, ax = plt.subplots()
    ax.pie(resumen.values, labels=[str(x) for x in resumen.index], autopct='%1.1f%%', startangle=90)
    ax.axis('equal'); ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    st.pyplot(fig); st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

def _choose_chart_auto(df: pd.DataFrame, cat_col: str, val_col: str) -> str:
    cat_norm = _norm(cat_col)
    id_hints = ["patente","folio","nro","numero","número","doc","documento","factura","boleta","oc","orden","presupuesto","cotizacion","cotización"]
    if any(h in cat_norm for h in id_hints): return "table"
    nunique = df[cat_col].nunique(dropna=False)
    if nunique <= 6: return "torta"
    if nunique <= st.session_state.get("max_cats_grafico", 18): return "barras"
    return "table"

# ---------------------------
# PARSER DE INSTRUCCIONES
# ---------------------------
VIZ_PATT = re.compile(r'(?:^|\n)\s*(?:viz\s*:\s*([^\n\r]+))', re.IGNORECASE)
ALT_PATT = re.compile(r'(grafico_torta|grafico_barras|tabla)(?:@([^\s:]+))?\s*:\s*([^\n\r]+)', re.IGNORECASE)

def _safe_plot(plot_fn, hoja, df, cat_raw, val_raw, titulo, cliente_txt):
    cat = find_col(df, cat_raw); val = find_col(df, val_raw)
    if not cat or not val:
        st.warning(f"❗ No se pudo generar en '{hoja}'. Revisar columnas: '{cat_raw}' y '{val_raw}'."); return
    try:
        plot_fn(df, cat, val, titulo)  # sin filtro por cliente
        st.session_state.aliases[_norm(cat_raw)] = cat
        st.session_state.aliases[_norm(val_raw)] = val
    except Exception as e:
        st.error(f"Error generando visualización en '{hoja}': {e}")

def parse_and_render_instructions(respuesta_texto: str, data_dict: dict, cliente_txt: str):
    # 1) nuevo formato viz:
    m = VIZ_PATT.search(respuesta_texto)
    if m:
        body = m.group(1).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]
        if len(parts) >= 3:
            tipo = parts[0].lower()
            cat_raw, val_raw = parts[1], parts[2]
            titulo = parts[3] if len(parts) >= 4 else None
            for hoja, df in data_dict.items():
                if find_col(df, cat_raw) and find_col(df, val_raw):
                    if tipo == "barras":   _safe_plot(mostrar_grafico_barras, hoja, df, cat_raw, val_raw, titulo, "")
                    elif tipo == "torta":  _safe_plot(mostrar_grafico_torta, hoja, df, cat_raw, val_raw, titulo, "")
                    else:                  _safe_plot(lambda d,a,b,t: mostrar_tabla(d,a,b,t or f"Tabla ({hoja})"), hoja, df, cat_raw, val_raw, titulo, "")
                    return

    # 2) compatibilidad con formatos antiguos
    for m in ALT_PATT.finditer(respuesta_texto):
        kind = m.group(1).lower()
        hoja_sel = m.group(2)
        body = m.group(3).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]
        if kind in ("grafico_torta","grafico_barras"):
            if len(parts) != 3:
                st.warning("Instrucción de gráfico inválida."); 
                continue
            cat_raw, val_raw, title = parts
            if hoja_sel and hoja_sel in data_dict:
                _safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                          hoja_sel, data_dict[hoja_sel], cat_raw, val_raw, title, "")
            else:
                for hoja, df in data_dict.items():
                    if find_col(df, cat_raw) and find_col(df, val_raw):
                        _safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                                  hoja, df, cat_raw, val_raw, title, "")
                        break
        else:
            if len(parts) not in (2,3):
                st.warning("Instrucción de tabla inválida.")
                continue
            cat_raw, val_raw = parts[0], parts[1]; title = parts[2] if len(parts)==3 else None
            def draw(df, hoja):
                cat = find_col(df, cat_raw); val = find_col(df, val_raw)
                if cat and val:
                    mostrar_tabla(df, cat, val, title or f"Tabla: {val} por {cat} ({hoja})")
                    st.session_state.aliases[_norm(cat_raw)] = cat
                    st.session_state.aliases[_norm(val_raw)] = val
                    return True
                return False
            if hoja_sel and hoja_sel in data_dict:
                draw(data_dict[hoja_sel], hoja_sel)
            else:
                for hoja, df in data_dict.items(): 
                    if draw(df, hoja): break

# ---------------------------
# SCHEMA + PLANNER + EXECUTOR
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
    system = "Eres un controller financiero experto. Sé concreto y útil."
    prompt = f"""
Devuelve SOLO un JSON con el mejor plan de visualización si corresponde:
{{
  "action": "table" | "chart" | "text",
  "sheet": "<nombre hoja o vacío>",
  "category_col": "<cat>",
  "value_col": "<val>",
  "chart": "barras" | "torta" | "linea" | "auto",
  "title": "<título>"
}}
Reglas:
- Usa nombres EXACTOS del esquema (insensible a mayúsculas).
- Si la categoría es un identificador (patente/folio/doc/factura/oc), prefiere "table".
- Si no se indica tipo de gráfico, usa "auto".
ESQUEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

PREGUNTA:
{pregunta}
"""
    messages = [{"role":"system","content":system}, {"role":"user","content":prompt}]
    raw = ask_gpt(messages).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m: return {}
    try:
        plan = json.loads(m.group(0)); return plan if isinstance(plan, dict) else {}
    except Exception:
        return {}

def execute_plan(plan: Dict[str, Any], data: Dict[str, Any], cliente_txt: str) -> bool:
    action = plan.get("action")
    if action not in ("table","chart","text"): return False
    if action == "text": return False
    sheet = plan.get("sheet") or ""
    cat = plan.get("category_col") or ""
    val = plan.get("value_col") or ""
    chart = (plan.get("chart") or "auto").lower()
    title = plan.get("title") or None

    hojas = [sheet] if sheet in data else list(data.keys())
    for h in hojas:
        df = data[h]
        if df is None or df.empty: continue
        cat_real = find_col(df, cat) if cat else None
        val_real = find_col(df, val) if val else None
        if not val_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["monto","importe","neto","total","facturacion","ingreso","venta","principal"]):
                    val_real = c; break
        if not cat_real:
            for c in df.columns:
                if any(k in _norm(c) for k in ["patente","cliente","tipo","estado","proceso","servicio","vehiculo","unidad","mes","fecha"]):
                    cat_real = c; break
        if not (cat_real and val_real): continue

        df_fil = df.copy()  # sin filtro por cliente

        if action == "chart" and chart == "auto":
            chart = _choose_chart_auto(df_fil, cat_real, val_real)

        if action == "table" or chart == "table":
            mostrar_tabla(df_fil, cat_real, val_real, title)
            st.session_state["__ultima_vista__"] = {"sheet": h, "cat": cat_real, "val": val_real, "type":"tabla"}
            return True

        if chart == "barras":   mostrar_grafico_barras(df_fil, cat_real, val_real, title)
        elif chart == "torta":  mostrar_grafico_torta(df_fil, cat_real, val_real, title)
        elif chart == "linea":  mostrar_grafico_barras(df_fil, cat_real, val_real, title)  # fallback
        else:                   mostrar_grafico_barras(df_fil, cat_real, val_real, title)

        st.session_state["__ultima_vista__"] = {"sheet": h, "cat": cat_real, "val": val_real, "type":f"chart:{chart}"}
        return True
    return False

# ---------------------------
# KPI DASHBOARD (GENÉRICO)
# ---------------------------
def _first_col_with(df: pd.DataFrame, keys):
    for c in df.columns:
        if any(k in _norm(str(c)) for k in keys):
            return c
    return None

def _ing_col(df: pd.DataFrame):
    for k in ["monto", "neto", "total", "importe", "facturacion", "ingreso", "venta", "principal"]:
        c = _first_col_with(df, [k])
        if c is not None: return c
    return None

def _date_col(df: pd.DataFrame):
    return _first_col_with(df, ["fecha", "mes", "emision", "emisión"])

def _count_col(df: pd.DataFrame):
    return _first_col_with(df, ["estado", "resultado", "situacion", "situación", "status"])

def render_kpi_dashboard(data: dict):
    st.markdown("#### 📊 Dashboard")
    c1, c2, c3 = st.columns(3)

    # 1) Top-10 Clientes (si existen columnas necesarias)
    plotted1 = False
    for hoja, df in data.items():
        cli = _first_col_with(df, ["cliente"])
        val = _ing_col(df)
        if cli and val:
            df2 = df.copy()
            vals = pd.to_numeric(df2[val], errors="coerce")
            top = (df2.assign(__v=vals).groupby(cli, dropna=False)["__v"]
                     .sum().sort_values(ascending=False).head(10))
            with c1:
                st.caption(f"Top-10 Clientes ({hoja})")
                _barras_horizontal(top, cli, val, titulo=None)
            plotted1 = True
            break
    if not plotted1:
        with c1: st.info("No se encontró columna de cliente para Top-10.")

    # 2) Distribución por Estado
    plotted2 = False
    for hoja, df in data.items():
        est = _count_col(df)
        if est:
            df2 = df.copy()
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

    # 3) Tendencia mensual (multi-hoja)
    ser_total = None
    for hoja, df in data.items():
        val = _ing_col(df); fec = _date_col(df)
        if val and fec:
            df2 = df.copy()
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
# IA – PROMPTS ESTRICTOS
# ---------------------------
def make_system_prompt():
    modo = st.session_state.get("modo_respuesta","Analítico (detallado)")
    tono = "breve, ejecutivo (5–7 frases)" if "Ejecutivo" in modo else "analítico, riguroso y accionable"
    return (
        "Actúas como un Controller Financiero senior especializado en talleres de desabolladura y pintura. "
        f"Tono {tono}. Responde siempre con secciones Markdown claras."
    )

ANALYSIS_FORMAT = """
Devuelve SIEMPRE en este formato Markdown (en español):

### Resumen ejecutivo
• 3–5 puntos clave con cifras redondeadas.

### Diagnóstico
• Qué está bien / mal y por qué (drivers).

### Estimaciones y proyecciones
• Proyección 3–6 meses (con supuestos explícitos).  
• Incluye rangos (optimista/base/conservador).

### Recomendaciones
• 5–8 acciones concretas y priorizadas (con impacto y dificultad).

### Riesgos y alertas
• 3–5 riesgos y cómo mitigarlos.

### Próximos pasos (dueños y fechas)
• Lista corta, muy específica.

Al final, si ayuda, añade UNA sola instrucción de visualización en alguna de estas formas (una línea):
viz: barras|<categoria>|<valor>|<título opcional>
viz: torta|<categoria>|<valor>|<título opcional>
viz: tabla|<categoria>|<valor>|<título opcional>
"""

def prompt_analisis_general(analisis_dict: dict) -> list:
    return [
        {"role":"system","content": make_system_prompt()},
        {"role":"user","content": f"""
Con base en los siguientes KPIs calculados reales, realiza un ANÁLISIS PROFESIONAL siguiendo el formato obligatorio.
No inventes datos fuera de lo entregado; si necesitas supuestos, decláralos.

KPIs:
{json.dumps(analisis_dict, ensure_ascii=False, indent=2)}

{ANALYSIS_FORMAT}
"""}
    ]

def prompt_consulta_libre(pregunta: str, schema: dict) -> list:
    historial_msgs = []
    for h in st.session_state.historial[-8:]:
        historial_msgs += [{"role":"user","content":h["pregunta"]},
                           {"role":"assistant","content":h["respuesta"]}]
    return [
        {"role":"system","content": make_system_prompt()},
        *historial_msgs,
        {"role":"user","content": f"""
Contesta la siguiente pregunta usando el esquema de datos; si corresponde incluye UNA instrucción de visualización.
Pregunta: {pregunta}

Esquema (hojas, columnas y ejemplos):
{json.dumps(schema, ensure_ascii=False, indent=2)}

{ANALYSIS_FORMAT}
"""}
    ]

def prompt_diagnostico_ia(contexto_cuant: dict) -> list:
    return [
        {"role":"system","content": make_system_prompt()},
        {"role":"user","content": f"""
Eres auditor financiero. Con el resumen numérico siguiente, emite diagnóstico, proyecciones y plan de acción.
{json.dumps(contexto_cuant, ensure_ascii=False, indent=2)}

{ANALYSIS_FORMAT}
"""}
    ]

# ---------------------------
# CÁLCULOS DE APOYO (DIAGNÓSTICO NUMÉRICO)
# ---------------------------
def _tendencia_mensual_global(data: dict):
    ser_total = None
    for _, df in data.items():
        val = _ing_col(df); fec = _date_col(df)
        if val and fec:
            df2 = df.copy()
            df2[fec] = pd.to_datetime(df2[fec], errors="coerce")
            df2["__v"] = pd.to_numeric(df2[val], errors="coerce")
            g = (df2.dropna(subset=[fec])
                    .set_index(fec)
                    .groupby(pd.Grouper(freq="M"))["__v"].sum())
            ser_total = g if ser_total is None else ser_total.add(g, fill_value=0)
    if ser_total is None or len(ser_total.dropna()) < 3:
        return None
    s = ser_total.dropna()
    return s

def _proyeccion_lineal(serie_mensual: pd.Series, meses=6):
    y = serie_mensual.values.astype(float)
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    pred = []
    for i in range(1, meses+1):
        pred.append(np.polyval(coef, len(y)-1 + i))
    return coef.tolist(), pred

def _concentracion_topn(df: pd.DataFrame, cliente_col: str, val_col: str, n=5):
    vals = pd.to_numeric(df[val_col], errors="coerce")
    dist = (df.assign(__v=vals).groupby(cliente_col, dropna=False)["__v"].sum())
    total = dist.sum()
    if total <= 0 or len(dist) == 0: return None
    share_top = dist.sort_values(ascending=False).head(n).sum() / total
    return float(share_top)

def _resumen_numerico(data: dict):
    kpis = analizar_datos_taller(data, "")  # sin filtro de cliente
    serie = _tendencia_mensual_global(data)
    proy = None; coef = None
    if serie is not None:
        coef, proy_vals = _proyeccion_lineal(serie, meses=6)
        proy = {
            "ultima_fecha": str(serie.index[-1].date()),
            "hist_len": int(len(serie)),
            "promedio_3m": float(serie.tail(3).mean()),
            "promedio_6m": float(serie.tail(min(6,len(serie))).mean()),
            "proyeccion_6m": [float(x) for x in proy_vals]
        }

    conc = None
    for _, df in data.items():
        cli = _first_col_with(df, ["cliente"])
        val = _ing_col(df)
        if cli and val:
            conc = _concentracion_topn(df, cli, val, n=5)
            if conc is not None: break

    return {
        "kpis": kpis,
        "tendencia_mensual": list(zip([str(x.date()) for x in serie.index], [float(v) for v in serie.values])) if serie is not None else None,
        "proyeccion_lineal": {"coef": coef, "detalles": proy} if proy else None,
        "concentracion_top5_clientes": conc
    }

# ---------------------------
# UI – SIDEBAR & PÁGINAS
# ---------------------------
st.title("🤖 Controller Financiero IA")

with st.sidebar:
    st.markdown("### Menú")
    st.session_state.menu_sel = st.radio(
        "Secciones",
        ["Datos","Vista previa","KPIs","Consulta IA","Historial","Diagnóstico IA"],
        index=["Datos","Vista previa","KPIs","Consulta IA","Historial","Diagnóstico IA"].index(st.session_state.menu_sel),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Preferencias")
    st.session_state.modo_respuesta = st.radio("Modo de respuesta", ["Ejecutivo (breve)","Analítico (detallado)"])
    st.session_state.max_cats_grafico = st.number_input("Máx. categorías para graficar", 6, 200, st.session_state.max_cats_grafico)
    st.session_state.top_n_grafico = st.number_input("Top-N por defecto (barras)", 5, 100, st.session_state.top_n_grafico)

# ----------- Datos -----------
if st.session_state.menu_sel == "Datos":
    st.markdown("### 📁 Datos")
    fuente = st.radio("Fuente", ["Excel","Google Sheets"], key="k_fuente")
    if fuente == "Excel":
        file = st.file_uploader("Sube un Excel", type=["xlsx","xls"])
        if file: 
            st.session_state.data = load_excel(file)
            st.success("Excel cargado.")
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

# ----------- Vista previa -----------
elif st.session_state.menu_sel == "Vista previa":
    data = st.session_state.data
    if not data:
        st.info("Carga datos en la sección **Datos**.")
    else:
        st.markdown("### 📄 Hojas")
        for name, df in data.items():
            st.markdown(f"#### 📘 {name} • filas: {len(df)}")
            st.dataframe(df.head(10), use_container_width=True)

# ----------- KPIs -----------
elif st.session_state.menu_sel == "KPIs":
    data = st.session_state.data
    if not data:
        st.info("Carga datos en la sección **Datos**.")
    else:
        kpis = analizar_datos_taller(data, "")  # sin filtro
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

        render_kpi_dashboard(data)

# ----------- Consulta IA -----------
elif st.session_state.menu_sel == "Consulta IA":
    data = st.session_state.data
    if not data:
        st.info("Carga datos en la sección **Datos**.")
    else:
        st.markdown("### 🤖 Consulta")
        pregunta = st.text_area("Pregunta")
        c1b, c2b = st.columns(2)
        if c1b.button("📊 Análisis General Automático"):
            analisis = analizar_datos_taller(data, "")
            r = ask_gpt(prompt_analisis_general(analisis))
            st.markdown(r)
            st.session_state.historial.append({"pregunta":"Análisis general","respuesta":r})
            parse_and_render_instructions(r, data, "")

        if c2b.button("Responder") and pregunta:
            schema = _build_schema(data)
            plan = plan_from_llm(pregunta, schema)
            executed = False
            if plan: executed = execute_plan(plan, data, "")
            if not executed:
                r = ask_gpt(prompt_consulta_libre(pregunta, schema))
                st.markdown(r)
                st.session_state.historial.append({"pregunta":pregunta,"respuesta":r})
                parse_and_render_instructions(r, data, "")

# ----------- Historial -----------
elif st.session_state.menu_sel == "Historial":
    if st.session_state.historial:
        for i, h in enumerate(st.session_state.historial[-20:], 1):
            st.markdown(f"**Q{i}:** {h['pregunta']}")
            st.markdown(f"**A{i}:** {h['respuesta']}")
    else:
        st.info("Aún no hay historial en esta sesión.")

# ----------- Diagnóstico IA -----------
elif st.session_state.menu_sel == "Diagnóstico IA":
    data = st.session_state.data
    if not data:
        st.info("Carga datos en la sección **Datos**.")
    else:
        st.markdown("### 🩺 Diagnóstico IA")
        ctx = _resumen_numerico(data)

        # Bloque numérico compacto (sin filtro por cliente)
        st.subheader("Resumen numérico")
        st.json(ctx)

        # Gráfico de tendencia si existe
        if ctx["tendencia_mensual"]:
            fechas = [pd.to_datetime(d) for d,_ in ctx["tendencia_mensual"]]
            valores = [v for _,v in ctx["tendencia_mensual"]]
            fig, ax = plt.subplots()
            ax.plot(fechas, valores, marker="o")
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
            fig.autofmt_xdate()
            st.pyplot(fig)

        # --- Nuevo: Diagnóstico de API OpenAI desde esta sección ---
        st.markdown("---")
        st.subheader("🔎 Diagnóstico de la IA (OpenAI)")
        if st.button("Diagnosticar IA"):
            diag = diagnosticar_openai()

            # Resultados claros
            st.write("**Clave configurada:** ", "✅" if diag["api_key_present"] else "❌")
            st.write("**Organization seteada:** ", "✅" if diag["organization_set"] else "—")
            st.write("**Base URL personalizada:** ", "✅" if diag["base_url_set"] else "—")
            st.write("**Listar modelos:** ", "✅" if diag["list_models_ok"] else "❌")
            st.write("**Prueba de chat:** ", "✅" if diag["chat_ok"] else "❌")
            if diag["quota_ok"] is True:
                st.success("Créditos/cuota: OK")
            elif diag["quota_ok"] is False:
                st.error("❌ Sin créditos/cuota (insufficient_quota). Carga saldo o revisa tu plan.")
            else:
                st.info("No se pudo determinar el estado de la cuota (revisa el mensaje de error).")

            if diag["usage_tokens"] is not None:
                st.caption(f"Tokens usados en la prueba: {diag['usage_tokens']}")

            if diag["error"]:
                st.warning(f"Detalle: {diag['error']}")

        if st.button("Generar diagnóstico con IA"):
            r = ask_gpt(prompt_diagnostico_ia(ctx))
            st.markdown(r)
            st.session_state.historial.append({"pregunta":"Diagnóstico IA","respuesta":r})
            parse_and_render_instructions(r, data, "")

