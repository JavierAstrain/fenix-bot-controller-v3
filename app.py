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
from typing import Dict, Any
from google.oauth2.service_account import Credentials
from openai import OpenAI
from analizador import analizar_datos_taller

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide", page_title="Controller Financiero IA")

# Tipografía/tamaño uniformes y desactivar cursivas
st.markdown("""
<style>
html, body, [data-testid="stMarkdownContainer"] {
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif !important;
  font-size: 15.5px !important;
  line-height: 1.55 !important;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  font-weight: 700 !important;
  letter-spacing: .2px;
}
/* Evita cursivas aunque el markdown incluya *...* o _..._ */
[data-testid="stMarkdownContainer"] em, 
[data-testid="stMarkdownContainer"] i { 
  font-style: normal !important; 
}
/* Evita monospace accidental en `code` */
[data-testid="stMarkdownContainer"] code { 
  font-family: inherit !important; 
  background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

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
# OPENAI
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
            temperature=0.15
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Fallo en la petición a OpenAI: {e}")
        return "⚠️ No pude completar la consulta a la IA."

def diagnosticar_openai():
    """Diagnóstico de credenciales/cuota."""
    res = {
        "api_key_present": False,
        "organization_set": False,
        "base_url_set": False,
        "list_models_ok": False,
        "chat_ok": False,
        "quota_ok": None,
        "usage_tokens": None,
        "error": None
    }
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

    client = _get_openai_client()
    if client is None:
        res["error"] = "No se pudo inicializar el cliente OpenAI."
        return res

    try:
        _ = client.models.list()
        res["list_models_ok"] = True
    except Exception as e:
        msg = str(e).lower()
        res["error"] = f"Error listando modelos: {e}"
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            res["quota_ok"] = False
        return res

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Ping de diagnóstico. Responde: OK."}],
            temperature=0,
            max_tokens=5
        )
        res["chat_ok"] = True
        try:
            if hasattr(r, "usage") and r.usage and r.usage.total_tokens is not None:
                res["usage_tokens"] = int(r.usage.total_tokens)
        except Exception:
            pass
        if res["quota_ok"] is None: res["quota_ok"] = True
    except Exception as e:
        msg = str(e).lower()
        res["error"] = f"Error en prueba de chat: {e}"
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            res["quota_ok"] = False
        else:
            res["quota_ok"] = None
    return res

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
# PARSER/HELPERS PARA TEXTO+VIZ
# ---------------------------
VIZ_PATT = re.compile(r'(?:^|\n)\s*(?:viz\s*:\s*([^\n\r]+))', re.IGNORECASE)
ALT_PATT = re.compile(r'(grafico_torta|grafico_barras|tabla)(?:@([^\s:]+))?\s*:\s*([^\n\r]+)', re.IGNORECASE)

# --- Normalización de texto y montos en respuestas IA ---
LETTER = r"A-Za-zÁÉÍÓÚÜÑáéíóúüñ"

# cantidades 1,234,567  y  1.234.567
CURRENCY_COMMA_RE = re.compile(r'(?<![\w$])(\d{1,3}(?:,\d{3})+)(?![\w%])')
CURRENCY_DOT_RE   = re.compile(r'(?<![\w$])(\d{1,3}(?:\.\d{3})+)(?![\w%])')

# series de cantidades separadas por coma/punto y/o "y/e/–/-"
LIST_COMMA_SERIES_RE = re.compile(
    r'(?<![\d$])((?:\d{1,3}(?:,\d{3})+)(?:\s*(?:,|y|e|–|-)\s*(?:\d{1,3}(?:,\d{3})+))+)(?![\d$])',
    re.I
)
LIST_DOT_SERIES_RE = re.compile(
    r'(?<![\d$])((?:\d{1,3}(?:\.\d{3})+)(?:\s*(?:,|y|e|–|-)\s*(?:\d{1,3}(?:\.\d{3})+))+)(?![\d$])',
    re.I
)

# "X millones" / "X mil" (con o sin $)
MILLONES_RE = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*millone?s\b')
MILES_RE    = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*mil\b')

# Conectores para despegar palabras
STOP = r"con|de|del|la|las|el|los|un|una|unos|unas|y|e|por|para|al|en|que|mientras|entre|sobre|hasta|desde"
GLUE_RE = re.compile(rf'([{LETTER}]{{2,}})({STOP})([{LETTER}]{{2,}})', re.I)
CAMEL_RE = re.compile(r'([a-záéíóúñ]{2,})([A-ZÁÉÍÓÚÑ])')  # antes de mayúscula pegada

INVISIBLES_RE = re.compile(r'[\u200B\u200C\u200D\uFEFF\u2060\u00AD]')  # ZWSP, ZWNJ, ZWJ, BOM, WORD JOINER, soft hyphen

def prettify_answer(text: str) -> str:
    """Limpia estilos/HTML, corrige pegados y normaliza montos (CLP)."""
    if not text:
        return text
    t = text

    # 0) limpiar HTML/estilos/char invisibles y bullets
    t = INVISIBLES_RE.sub('', t)
    t = re.sub(r'<[^>]+>', '', t)                       # tags HTML
    t = t.replace("\u00A0", " ")                        # NBSP
    t = t.replace("•", "\n- ")                          # bullets
    t = t.replace("*", "").replace("_", "").replace("`", "")  # quita énfasis markdown
    t = t.replace("“","\"").replace("”","\"").replace("’","'")

    # 1) separar dígitos/letras
    t = re.sub(rf'(?<=\d)(?=[{LETTER}])', ' ', t)
    t = re.sub(rf'(?<=[{LETTER}])(?=\d)', ' ', t)

    # 2) despegar conectores y mayúsculas pegadas (loop hasta estabilizar)
    prev = None
    while prev != t:
        prev = t
        t = GLUE_RE.sub(r'\1 \2 \3', t)
        t = CAMEL_RE.sub(r'\1 \2', t)

    # 3) normaliza inicio de bullet y espacios
    t = re.sub(r'^[\s]*[-•]\s*', '- ', t, flags=re.M)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n\s+', '\n', t)

    # 4) "X millones" y "X mil" -> CLP
    def _mill_to_clp(m):
        raw = m.group(1).replace(",", ".")
        try:
            val = float(raw) * 1_000_000
            return f"${int(round(val)):,}".replace(",", ".")
        except Exception:
            return m.group(0)
    t = MILLONES_RE.sub(_mill_to_clp, t)

    def _mil_to_clp(m):
        raw = m.group(1).replace(",", ".")
        try:
            val = float(raw) * 1_000
            return f"${int(round(val)):,}".replace(",", ".")
        except Exception:
            return m.group(0)
    t = MILES_RE.sub(_mil_to_clp, t)

    # 5) series de cantidades -> $ y "–"
    def _series_to_clp_comma(m):
        nums = re.findall(r'\d{1,3}(?:,\d{3})+', m.group(1))
        clp = [f"${int(n.replace(',', '')):,}".replace(",", ".") for n in nums]
        return " – ".join(clp)
    t = LIST_COMMA_SERIES_RE.sub(_series_to_clp_comma, t)

    def _series_to_clp_dot(m):
        nums = re.findall(r'\d{1,3}(?:\.\d{3})+', m.group(1))
        clp = [f"${int(n.replace('.', '')):,}".replace(",", ".") for n in nums]
        return " – ".join(clp)
    t = LIST_DOT_SERIES_RE.sub(_series_to_clp_dot, t)

    # 6) cantidades sueltas con separador -> CLP
    t = CURRENCY_COMMA_RE.sub(lambda m: f"${int(m.group(1).replace(',', '')):,}".replace(",", "."), t)
    t = CURRENCY_DOT_RE.sub(  lambda m: f"${int(m.group(1).replace('.', '')):,}".replace(",", "."), t)

    # 7) normalizar prefijos y doble $
    t = re.sub(r'(?:US|CLP|S|COP)\$', '$', t, flags=re.I)  # US$, CLP$, S$, COP$ -> $
    t = re.sub(r'\${2,}', '$', t)                          # $$ -> $

    # 8) espacios tras puntuación
    t = re.sub(r':(?=\S)', ': ', t)
    t = re.sub(r',(?=\S)', ', ', t)

    return t.strip()

def split_text_and_viz(respuesta_texto: str):
    """Devuelve (texto_sin_viz, lista_instrucciones)."""
    text = respuesta_texto
    instr = []
    for m in VIZ_PATT.finditer(respuesta_texto):
        body = m.group(1).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]
        if len(parts) >= 3:
            instr.append(("viz", parts))
        text = text.replace(m.group(0), "")
    for m in ALT_PATT.finditer(respuesta_texto):
        kind = m.group(1).lower()
        hoja = m.group(2)
        body = m.group(3).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]
        instr.append((kind, hoja, parts))
        text = text.replace(m.group(0), "")
    return text.strip(), instr

def _safe_plot(plot_fn, hoja, df, cat_raw, val_raw, titulo):
    cat = find_col(df, cat_raw); val = find_col(df, val_raw)
    if not cat or not val:
        st.warning(f"❗ No se pudo generar en '{hoja}'. Revisar columnas: '{cat_raw}' y '{val_raw}'."); return
    try:
        plot_fn(df, cat, val, titulo)
        st.session_state.aliases[_norm(cat_raw)] = cat
        st.session_state.aliases[_norm(val_raw)] = val
    except Exception as e:
        st.error(f"Error generando visualización en '{hoja}': {e}")

def render_viz_instructions(instr_list, data_dict):
    """Ejecuta la primera instrucción que haga match con alguna hoja."""
    if not instr_list: return False
    for item in instr_list:
        if item[0] == "viz":
            _tipo, parts = item
            tipo = parts[0].lower()
            cat_raw, val_raw = parts[1], parts[2]
            titulo = parts[3] if len(parts) >= 4 else None
            for hoja, df in data_dict.items():
                if find_col(df, cat_raw) and find_col(df, val_raw):
                    if tipo == "barras":   _safe_plot(mostrar_grafico_barras, hoja, df, cat_raw, val_raw, titulo); return True
                    if tipo == "torta":    _safe_plot(mostrar_grafico_torta, hoja, df, cat_raw, val_raw, titulo); return True
                    _safe_plot(lambda d,a,b,t: mostrar_tabla(d,a,b,t or f"Tabla ({hoja})"), hoja, df, cat_raw, val_raw, titulo); return True
        else:
            kind, hoja_sel, parts = item
            if kind in ("grafico_torta","grafico_barras"):
                if len(parts) != 3: continue
                cat_raw, val_raw, title = parts
                if hoja_sel and hoja_sel in data_dict:
                    _safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                              hoja_sel, data_dict[hoja_sel], cat_raw, val_raw, title); return True
                for hoja, df in data_dict.items():
                    if find_col(df, cat_raw) and find_col(df, val_raw):
                        _safe_plot(mostrar_grafico_torta if kind=="grafico_torta" else mostrar_grafico_barras,
                                  hoja, df, cat_raw, val_raw, title); return True
            else:
                if len(parts) not in (2,3): continue
                cat_raw, val_raw = parts[0], parts[1]; title = parts[2] if len(parts)==3 else None
                for hoja, df in data_dict.items():
                    if find_col(df, cat_raw) and find_col(df, val_raw):
                        mostrar_tabla(df, find_col(df, cat_raw), find_col(df, val_raw), title); return True
    return False

# ---------------------------
# SCHEMA + PLANNER (fallback de viz)
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

def execute_plan(plan: Dict[str, Any], data: Dict[str, Any]) -> bool:
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

        df_fil = df.copy()

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
# IA – PROMPTS ESTRICTOS
# ---------------------------
def make_system_prompt():
    return (
        "Eres un Controller Financiero senior para un taller de desabolladura y pintura. "
        "Responde SIEMPRE con texto claro en Markdown, sin bloques de código, sin tipografías especiales ni emojis. "
        "Tu estilo combina: ejecutivo (conclusiones claras) + analítico (fundamentos y números). "
        "Toda la respuesta debe basarse EXCLUSIVAMENTE en los datos entregados por la planilla; "
        "si un dato no está, indica 'No disponible en planilla'. No uses conocimiento externo."
    )

ANALYSIS_FORMAT = """
Escribe SIEMPRE en este formato (solo Markdown simple y guiones '-' en bullets):

### Resumen ejecutivo
- 3 a 5 puntos clave con cifras redondeadas y contexto.

### Diagnóstico
- Qué está bien / mal y por qué (drivers por línea de negocio, clientes, estados, tiempos, etc.).

### Estimaciones y proyecciones
- Proyección 3–6 meses con supuestos explícitos y tres escenarios: optimista, base y conservador.

### Recomendaciones de gestión
- 5–8 acciones priorizadas (impacto estimado y dificultad operacional).

### Riesgos y alertas
- 3–5 riesgos y mitigaciones.

### Próximos pasos
- Dueños, plazos y métrica objetivo.

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
No inventes datos fuera de lo entregado; si necesitas supuestos, decláralos de forma conservadora.

KPIs:
{json.dumps(analisis_dict, ensure_ascii=False, indent=2)}

{ANALYSIS_FORMAT}
"""}
    ]

def prompt_consulta_libre(pregunta: str, schema: dict) -> list:
    historial_msgs = []
    for h in st.session_state.historial[-6:]:
        historial_msgs += [{"role":"user","content":h["pregunta"]},
                           {"role":"assistant","content":h["respuesta"]}]
    return [
        {"role":"system","content": make_system_prompt()},
        *historial_msgs,
        {"role":"user","content": f"""
Contesta la siguiente pregunta usando el esquema de datos; SIEMPRE incluye el análisis completo del formato y, si aplica, UNA instrucción de visualización.
Pregunta: {pregunta}

Esquema (hojas, columnas y ejemplos):
{json.dumps(schema, ensure_ascii=False, indent=2)}

{ANALYSIS_FORMAT}
"""}
    ]

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

        st.markdown("#### 📊 Dashboard")
        c1d, c2d, c3d = st.columns(3)

        # Top-10 clientes
        plotted1 = False
        for hoja, df in data.items():
            cli = next((c for c in df.columns if "cliente" in _norm(c)), None)
            val = next((c for c in df.columns if any(k in _norm(c) for k in ["monto","neto","total","importe","facturacion","ingreso","venta","principal"])), None)
            if cli and val:
                vals = pd.to_numeric(df[val], errors="coerce")
                top = (df.assign(__v=vals).groupby(cli, dropna=False)["__v"]
                         .sum().sort_values(ascending=False).head(10))
                with c1d:
                    st.caption(f"Top-10 Clientes ({hoja})")
                    _barras_horizontal(top, cli, val, titulo=None)
                plotted1 = True
                break
        if not plotted1:
            with c1d: st.info("No se encontró columna de cliente para Top-10.")

        # Estados
        plotted2 = False
        for hoja, df in data.items():
            est = next((c for c in df.columns if any(k in _norm(c) for k in ["estado","resultado","situacion","situación","status"])), None)
            if est:
                dist = df[est].astype(str).str.strip().str.title().value_counts().head(8)
                if len(dist) >= 1:
                    with c2d:
                        st.caption(f"Distribución por Estado ({hoja})")
                        fig, ax = plt.subplots()
                        ax.pie(dist.values, labels=dist.index.astype(str), autopct="%1.0f%%", startangle=90)
                        ax.axis("equal")
                        st.pyplot(fig)
                        st.download_button("⬇️ PNG", _export_fig(fig), "kpi_estado.png", "image/png")
                    plotted2 = True
                    break
        if not plotted2:
            with c2d: st.info("No se encontró columna de estado para distribución.")

        # Tendencia mensual
        ser_total = None
        for hoja, df in data.items():
            val = next((c for c in df.columns if any(k in _norm(c) for k in ["monto","neto","total","importe","facturacion","ingreso","venta","principal"])), None)
            fec = next((c for c in df.columns if any(k in _norm(c) for k in ["fecha","mes","emision","emisión"])), None)
            if val and fec:
                df2 = df.copy()
                df2[fec] = pd.to_datetime(df2[fec], errors="coerce")
                df2["__v"] = pd.to_numeric(df2[val], errors="coerce")
                g = (df2.dropna(subset=[fec]).set_index(fec).groupby(pd.Grouper(freq="M"))["__v"].sum())
                ser_total = g if ser_total is None else ser_total.add(g, fill_value=0)
        if ser_total is not None and len(ser_total.dropna()) >= 2:
            with c3d:
                st.caption("Tendencia Mensual de Ingresos (todas las hojas con fecha)")
                fig, ax = plt.subplots()
                ax.plot(ser_total.index, ser_total.values, marker="o")
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
                fig.autofmt_xdate()
                st.pyplot(fig)
                st.download_button("⬇️ PNG", _export_fig(fig), "kpi_tendencia.png", "image/png")
        else:
            with c3d: st.info("No se detectaron fechas para tendencia mensual.")

# ----------- Consulta IA (texto + visual)
elif st.session_state.menu_sel == "Consulta IA":
    data = st.session_state.data
    if not data:
        st.info("Carga datos en la sección **Datos**.")
    else:
        st.markdown("### 🤖 Consulta")
        pregunta = st.text_area("Pregunta")
        c1b, c2b = st.columns(2)

        left, right = st.columns([0.58, 0.42])

        if c1b.button("📊 Análisis General Automático"):
            analisis = analizar_datos_taller(data, "")
            raw = ask_gpt(prompt_analisis_general(analisis))
            texto, instr = split_text_and_viz(raw)
            texto = prettify_answer(texto)
            with left:
                st.markdown(texto)
            with right:
                ok = render_viz_instructions(instr, data)
                if not ok:
                    schema = _build_schema(data)
                    plan = plan_from_llm("Gráfico sugerido según KPIs", schema)
                    execute_plan(plan, data)
            st.session_state.historial.append({"pregunta":"Análisis general","respuesta":texto})

        if c2b.button("Responder") and pregunta:
            schema = _build_schema(data)
            raw = ask_gpt(prompt_consulta_libre(pregunta, schema))
            texto, instr = split_text_and_viz(raw)
            texto = prettify_answer(texto)
            with left:
                st.markdown(texto)
            with right:
                ok = render_viz_instructions(instr, data)
                if not ok:
                    plan = plan_from_llm(pregunta, schema)
                    execute_plan(plan, data)
            st.session_state.historial.append({"pregunta":pregunta,"respuesta":texto})

# ----------- Historial -----------
elif st.session_state.menu_sel == "Historial":
    if st.session_state.historial:
        for i, h in enumerate(st.session_state.historial[-20:], 1):
            st.markdown(f"**Q{i}:** {h['pregunta']}")
            st.markdown(f"**A{i}:**")
            st.markdown(h["respuesta"])
    else:
        st.info("Aún no hay historial en esta sesión.")

# ----------- Diagnóstico IA -----------
elif st.session_state.menu_sel == "Diagnóstico IA":
    st.markdown("### 🔎 Diagnóstico de la IA (OpenAI)")
    st.caption("Verifica API Key, conexión, prueba mínima de chat y estado de créditos/cuota.")
    if st.button("Diagnosticar IA"):
        diag = diagnosticar_openai()
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
            st.info("No se pudo determinar el estado de la cuota (revisa el mensaje).")
        if diag["usage_tokens"] is not None:
            st.caption(f"Tokens usados en la prueba: {diag['usage_tokens']}")
        if diag["error"]:
            st.warning(f"Detalle: {diag['error']}")
