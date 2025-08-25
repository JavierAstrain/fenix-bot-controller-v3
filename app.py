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
from typing import Dict, Any, List, Optional
from google.oauth2.service_account import Credentials
from openai import OpenAI
from analizador import analizar_datos_taller

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide", page_title="Controller Financiero IA")

# Tipografía/tamaño uniformes + neutralización de cursivas/monospace
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
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] i { font-style: normal !important; }
[data-testid="stMarkdownContainer"] code { font-family: inherit !important; background: transparent !important; }
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
        return resp.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Fallo en la petición a OpenAI: {e}")
        return "⚠️ No pude completar la consulta a la IA."

# Estructurado (JSON)
def ask_gpt_structured(messages) -> Optional[dict]:
    """Devuelve dict con la estructura esperada o None si no se pudo parsear."""
    raw = ask_gpt(messages).strip()
    # Extrae el primer bloque JSON “grande”
    m = re.search(r"\{(?:[^{}]|(?R))*\}", raw, flags=re.S)
    if not m:
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.S|re.I)
        if m: raw = m.group(1)
        else: return None
    else:
        raw = m.group(0)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def diagnosticar_openai():
    res = {
        "api_key_present": False, "organization_set": False, "base_url_set": False,
        "list_models_ok": False, "chat_ok": False, "quota_ok": None,
        "usage_tokens": None, "error": None
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
        _ = client.models.list(); res["list_models_ok"] = True
    except Exception as e:
        res["error"] = f"Error listando modelos: {e}"; return res
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Ping diagnóstico. Responde: OK."}],
            temperature=0, max_tokens=5
        )
        res["chat_ok"] = True
        if getattr(r, "usage", None) and getattr(r.usage, "total_tokens", None) is not None:
            res["usage_tokens"] = int(r.usage.total_tokens)
        if res["quota_ok"] is None: res["quota_ok"] = True
    except Exception as e:
        res["error"] = f"Error en prueba de chat: {e}"
        res["quota_ok"] = None
    return res

# ---------------------------
# UTILIDADES – Sanitización fuerte
# ---------------------------
LETTER = r"A-Za-zÁÉÍÓÚÜÑáéíóúüñ"

# Separadores de miles
CURRENCY_COMMA_RE = re.compile(r'(?<![\w$])(\d{1,3}(?:,\d{3})+)(?![\w%])')
CURRENCY_DOT_RE   = re.compile(r'(?<![\w$])(\d{1,3}(?:\.\d{3})+)(?![\w%])')

# Series
LIST_COMMA_SERIES_RE = re.compile(
    r'(?<![\d$])((?:\d{1,3}(?:,\d{3})+)(?:\s*(?:,|y|e|–|-)\s*(?:\d{1,3}(?:,\d{3})+))+)(?![\d$])',
    re.I
)
LIST_DOT_SERIES_RE = re.compile(
    r'(?<![\d$])((?:\d{1,3}(?:\.\d{3})+)(?:\s*(?:,|y|e|–|-)\s*(?:\d{1,3}(?:\.\d{3})+))+)(?![\d$])',
    re.I
)

MILLONES_RE = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*millone?s\b')
MILES_RE    = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*mil\b')

CONNECTORS = ["con","de","del","la","las","el","los","un","una","unos","unas","y","e",
              "por","para","al","en","que","mientras","entre","sobre","hasta","desde"]

ALL_SPACES_RE = re.compile(r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]')
INVISIBLES_RE = re.compile(r'[\u200B\u200C\u200D\uFEFF\u2060\u00AD]')
TITLE_LINE_RE = re.compile(r'^(#{1,6}\s+[^\n]+)$', re.M)

def _to_clp(i: int) -> str:
    return f"${i:,}".replace(",", ".")

def _inject_spaces_connectors(t: str) -> str:
    # Espacios alrededor de conectores cuando están pegados a letras
    for w in CONNECTORS:
        # inicio/fin o pegado a letras (sin romper palabras como 'conjunto' → no se toca)
        t = re.sub(rf'(?i)(?<=\b){w}(?=[{LETTER}])', f'{w} ', t)  # "conalgo" -> "con algo"
        t = re.sub(rf'(?i)(?<=[{LETTER}]){w}(?=\b)', f' {w}', t)  # "algocon" -> "algo con"
    # Fixes comunes que aparecen en finanzas
    t = re.sub(r'(?i)\bmargende\b', 'margen de', t)
    t = re.sub(r'(?i)\bmientrasque\b', 'mientras que', t)
    return t

def prettify_answer(text: str) -> str:
    """Saneado general + normalización financiera (CLP)."""
    if not text: return text
    t = text

    # (0) NFKC: mata letras estilizadas (𝑨, 𝙵, etc.)
    t = unicodedata.normalize("NFKC", t)

    # (1) tags HTML, invisibles y espacios exóticos
    t = re.sub(r'<[^>]+>', '', t)
    t = INVISIBLES_RE.sub('', t)
    t = ALL_SPACES_RE.sub(' ', t)
    t = t.replace("•", "\n- ")
    t = t.replace("“","\"").replace("”","\"").replace("’","'")

    # (2) neutralizar énfasis/código markdown
    t = re.sub(r'([*_`~]{1,3})(?=\S)(.+?)(?<=\S)\1', r'\2', t)

    # (3) separaciones básicas
    t = TITLE_LINE_RE.sub(r'\1\n', t)
    t = re.sub(r'^[\s]*[-•]\s*', '- ', t, flags=re.M)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n\s+', '\n', t)

    # (4) inyectar espacios en conectores pegados
    t = _inject_spaces_connectors(t)
    # minúscula seguida de MAYÚSCULA pegada
    t = re.sub(r'([a-záéíóúñ]{2,})([A-ZÁÉÍÓÚÑ])', r'\1 \2', t)

    # (5) normalizar prefijos monetarios ANTES de convertir números
    t = re.sub(r'(?i)(?:US|CLP|COP|S)\s*\$', '$', t)

    # (6) millones / mil → CLP
    def _mill_to_clp(m):
        raw = m.group(1).replace(",", ".")
        try: return _to_clp(int(round(float(raw) * 1_000_000)))
        except: return m.group(0)
    t = MILLONES_RE.sub(_mill_to_clp, t)

    def _mil_to_clp(m):
        raw = m.group(1).replace(",", ".")
        try: return _to_clp(int(round(float(raw) * 1_000)))
        except: return m.group(0)
    t = MILES_RE.sub(_mil_to_clp, t)

    # (7) series
    def _series(nums, dot=False):
        out = []
        for n in nums:
            val = int(n.replace('.' if dot else ',', ''))
            out.append(_to_clp(val))
        return " – ".join(out)
    t = LIST_COMMA_SERIES_RE.sub(lambda m: _series(re.findall(r'\d{1,3}(?:,\d{3})+', m.group(1)), False), t)
    t = LIST_DOT_SERIES_RE.sub(  lambda m: _series(re.findall(r'\d{1,3}(?:\.\d{3})+', m.group(1)), True),  t)

    # (8) cantidades sueltas con separador
    t = CURRENCY_COMMA_RE.sub(lambda m: _to_clp(int(m.group(1).replace(',', ''))), t)
    t = CURRENCY_DOT_RE.sub(  lambda m: _to_clp(int(m.group(1).replace('.', ''))), t)

    # (9) colapsos de $
    t = re.sub(r'\$\s*\$+', '$', t)
    t = re.sub(r'\$\s+(?=\d)', '$', t)

    # (10) espacios tras puntuación
    t = re.sub(r':(?=\S)', ': ', t)
    t = re.sub(r',(?=\S)', ', ', t)

    return t.strip()

# ---------------------------
# VISUALIZACIONES
# ---------------------------
def _fmt_pesos(x, pos=None):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return ""
        return f"${int(round(x)):,}".replace(",", ".")
    except Exception: return str(x)

def _export_fig(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=180); buf.seek(0); return buf.read()

def find_col(df: pd.DataFrame, name: str):
    if not name: return None
    alias = st.session_state.aliases.get(re.sub(r'\s+',' ', unicodedata.normalize('NFD', name).lower()))
    if alias and alias in df.columns: return alias
    tgt = re.sub(r'\s+',' ', unicodedata.normalize('NFD', name).lower())
    for c in df.columns:
        if re.sub(r'\s+',' ', unicodedata.normalize('NFD', c).lower()) == tgt: return c
    candidates = [c for c in df.columns if tgt in re.sub(r'\s+',' ', unicodedata.normalize('NFD', c).lower())]
    return candidates[0] if candidates else None

def mostrar_tabla(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"].sum().sort_values(ascending=False).reset_index())
    resumen.columns = [str(col_categoria).title(), str(col_valor).title()]
    col_val = resumen.columns[1]
    resumen[col_val] = resumen[col_val].apply(_fmt_pesos)
    total_val = vals.sum(skipna=True)
    resumen.loc[len(resumen)] = ["TOTAL", _fmt_pesos(total_val)]
    st.markdown(f"### 📊 {titulo if titulo else f'{col_val} por {col_categoria}'}")
    st.dataframe(resumen, use_container_width=True)
    st.download_button("⬇️ Descargar tabla (CSV)", resumen.to_csv(index=False).encode("utf-8"), "tabla.csv", "text/csv")

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
            ax.annotate(_fmt_pesos(y), (b.get_x()+b.get_width()/2, y), textcoords="offset points", xytext=(0,3), ha='center', va='bottom', fontsize=8)
    fig.tight_layout(); st.pyplot(fig); st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

def _barras_horizontal(resumen, col_categoria, col_valor, titulo):
    fig, ax = plt.subplots()
    bars = ax.barh(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    ax.set_xlabel(col_valor); ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    for b in bars:
        x = b.get_width()
        if np.isfinite(x):
            ax.annotate(_fmt_pesos(x), (x, b.get_y()+b.get_height()/2), textcoords="offset points", xytext=(5,0), ha='left', va='center', fontsize=8)
    fig.tight_layout(); st.pyplot(fig); st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

def mostrar_grafico_barras(df, col_categoria, col_valor, titulo=None, top_n=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"].sum().sort_values(ascending=False))
    try:
        if len(resumen) >= 8:
            top_val = float(resumen.iloc[0]); med = float(np.median(resumen.values))
            if med > 0 and top_val / med >= 3.0:
                st.info("Distribución desbalanceada: muestro tabla para mejor lectura.")
                mostrar_tabla(df, col_categoria, col_valor, titulo); return
    except Exception: pass
    if top_n is None: top_n = st.session_state.get("top_n_grafico", 12)
    recorte = False
    if len(resumen) > top_n: resumen = resumen.head(top_n); recorte = True
    labels = resumen.index.astype(str)
    if labels.str.len().mean() > 10: _barras_horizontal(resumen, col_categoria, col_valor, titulo)
    else:                            _barras_vertical(resumen, col_categoria, col_valor, titulo)
    if recorte: st.caption(f"Mostrando Top-{top_n}. Usa tabla para el detalle completo.")

def mostrar_grafico_torta(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"].sum().sort_values(ascending=False))
    fig, ax = plt.subplots()
    ax.pie(resumen.values, labels=[str(x) for x in resumen.index], autopct='%1.1f%%', startangle=90)
    ax.axis('equal'); ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    st.pyplot(fig); st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

def _choose_chart_auto(df: pd.DataFrame, cat_col: str, val_col: str) -> str:
    norm = lambda s: re.sub(r'\s+',' ', unicodedata.normalize('NFD', s).lower())
    cat_norm = norm(cat_col)
    id_hints = ["patente","folio","nro","numero","número","doc","documento","factura","boleta","oc","orden","presupuesto","cotizacion","cotización"]
    if any(h in cat_norm for h in id_hints): return "table"
    nunique = df[cat_col].nunique(dropna=False)
    if nunique <= 6: return "torta"
    if nunique <= st.session_state.get("max_cats_grafico", 18): return "barras"
    return "table"

# ---------------------------
# TEXTO+VIZ (parser simple)
# ---------------------------
VIZ_PATT = re.compile(r'(?:^|\n)\s*(?:viz\s*:\s*([^\n\r]+))', re.IGNORECASE)

def split_text_and_viz(respuesta_texto: str):
    text = respuesta_texto
    instr = []
    for m in VIZ_PATT.finditer(respuesta_texto):
        body = m.group(1).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]
        if len(parts) >= 3: instr.append(("viz", parts))
        text = text.replace(m.group(0), "")
    return text.strip(), instr

def _safe_plot(plot_fn, hoja, df, cat_raw, val_raw, titulo):
    cat = find_col(df, cat_raw); val = find_col(df, val_raw)
    if not cat or not val:
        st.warning(f"❗ No se pudo generar en '{hoja}'. Revisar columnas: '{cat_raw}' y '{val_raw}'."); return
    try:
        plot_fn(df, cat, val, titulo)
        st.session_state.aliases[cat_raw.lower()] = cat
        st.session_state.aliases[val_raw.lower()] = val
    except Exception as e:
        st.error(f"Error generando visualización en '{hoja}': {e}")

def render_viz_instructions(instr_list, data_dict):
    if not instr_list: return False
    for item in instr_list:
        _tipo, parts = item
        tipo = parts[0].lower(); cat_raw, val_raw = parts[1], parts[2]
        titulo = parts[3] if len(parts) >= 4 else None
        for hoja, df in data_dict.items():
            if find_col(df, cat_raw) and find_col(df, val_raw):
                if tipo == "barras":   _safe_plot(mostrar_grafico_barras, hoja, df, cat_raw, val_raw, titulo); return True
                if tipo == "torta":    _safe_plot(mostrar_grafico_torta, hoja, df, cat_raw, val_raw, titulo); return True
                mostrar_tabla(df, find_col(df, cat_raw), find_col(df, val_raw), titulo or f"Tabla ({hoja})"); return True
    return False

# ---------------------------
# ESTRUCTURADO: prompts y render
# ---------------------------
def make_system_prompt():
    return (
        "Eres un Controller Financiero senior para un taller de desabolladura y pintura. "
        "Responde SIEMPRE basándote EXCLUSIVAMENTE en la planilla proporcionada."
    )

STRUCT_SCHEMA = {
    "resumen_ejecutivo": [],
    "diagnostico": [],
    "estimaciones_proyecciones": [],
    "recomendaciones": [],
    "riesgos_alertas": [],
    "proximos_pasos": [],
    "viz": None  # ejemplo: "barras|Categoria|Monto|Título opcional"
}

def prompt_analisis_general_JSON(analisis_dict: dict) -> list:
    return [
        {"role":"system","content": make_system_prompt()},
        {"role":"user","content": f"""
Genera SOLO un JSON UTF-8 sin Markdown ni texto fuera del JSON, con las claves:
{json.dumps(STRUCT_SCHEMA, ensure_ascii=False, indent=2)}

Reglas:
- Usa español neutro, claro y conciso.
- Usa solo caracteres ASCII básicos y acentos normales (no letras itálicas/estilizadas, no NBSP).
- En montos, idealmente CLP con separador de miles punto (ej. $2.160.000). Si dudas, escribe números enteros y yo los formateo.
- Si no hay datos, escribe "No disponible en planilla" en la lista correspondiente.
- Campo "viz": opcional, formato "barras|<categoria>|<valor>|<título>".

KPIs calculados:
{json.dumps(analisis_dict, ensure_ascii=False, indent=2)}
"""}
    ]

def prompt_consulta_libre_JSON(pregunta: str, schema: dict) -> list:
    return [
        {"role":"system","content": make_system_prompt()},
        {"role":"user","content": f"""
Con la siguiente pregunta y el esquema de datos, responde SOLO en JSON con las claves de:
{json.dumps(STRUCT_SCHEMA, ensure_ascii=False, indent=2)}

- Nada de Markdown fuera del JSON.
- Usa texto simple (ASCII normal). Evita emojis y letras estilizadas.
- Si puedes, agrega "viz" como "barras|<categoria>|<valor>|<título>".

Pregunta: {pregunta}

Esquema de hojas/columnas (no inventes):
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""}
    ]

def render_structured(obj: dict, data: Dict[str, pd.DataFrame]):
    # Sanitiza cada bullet y renderiza en columnas
    def _bul(section: str) -> List[str]:
        arr = obj.get(section) or []
        if not isinstance(arr, list): return []
        clean = [prettify_answer(str(x)) for x in arr if str(x).strip()]
        return clean

    left, right = st.columns([0.58, 0.42])

    with left:
        sec = [
            ("Resumen ejecutivo", _bul("resumen_ejecutivo")),
            ("Diagnóstico", _bul("diagnostico")),
            ("Estimaciones y proyecciones", _bul("estimaciones_proyecciones")),
            ("Recomendaciones de gestión", _bul("recomendaciones")),
            ("Riesgos y alertas", _bul("riesgos_alertas")),
            ("Próximos pasos", _bul("proximos_pasos")),
        ]
        for title, items in sec:
            if items:
                st.subheader(title)
                st.markdown("\n".join(f"- {i}" for i in items))

    viz = obj.get("viz")
    if viz and isinstance(viz, str):
        parts = [p.strip() for p in viz.split("|")]
        if len(parts) >= 3:
            kind = parts[0].lower(); cat_raw, val_raw = parts[1], parts[2]
            title = parts[3] if len(parts) >= 4 else None
            with right:
                ok = False
                for hoja, df in data.items():
                    if find_col(df, cat_raw) and find_col(df, val_raw):
                        if kind == "barras":   _safe_plot(mostrar_grafico_barras, hoja, df, cat_raw, val_raw, title); ok = True; break
                        if kind == "torta":    _safe_plot(mostrar_grafico_torta, hoja, df, cat_raw, val_raw, title); ok = True; break
                        mostrar_tabla(df, find_col(df, cat_raw), find_col(df, val_raw), title or f"Tabla ({hoja})"); ok = True; break
                if not ok:
                    st.info("No se encontró una combinación de columnas para la visualización propuesta.")

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
{{ "action":"chart|table|text", "sheet":"", "category_col":"", "value_col":"", "chart":"barras|torta|auto", "title":"" }}
Reglas: usa nombres EXACTOS del esquema (insensible a mayúsculas). Si no procede, "action":"text".
ESQUEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}
PREGUNTA:
{pregunta}
"""
    messages = [{"role":"system","content":system}, {"role":"user","content":prompt}]
    raw = ask_gpt(messages).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except Exception: return {}

def execute_plan(plan: Dict[str, Any], data: Dict[str, Any]) -> bool:
    action = (plan or {}).get("action")
    if action not in ("table","chart"): return False
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
        if not (cat_real and val_real): continue
        if action == "table": mostrar_tabla(df, cat_real, val_real, title); return True
        if chart in ("auto","barras"): mostrar_grafico_barras(df, cat_real, val_real, title); return True
        if chart == "torta": mostrar_grafico_torta(df, cat_real, val_real, title); return True
    return False

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
        kpis = analizar_datos_taller(data, "")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ingresos ($)", f"{int(round(kpis['ingresos'])):,}".replace(",", "."))
        c2.metric("Costos ($)",   f"{int(round(kpis['costos'])):,}".replace(",", "."))
        c3.metric("Margen ($)",   f"{int(round(kpis['margen'])):,}".replace(",", "."))
        c4.metric("Margen %",     f"{(kpis.get('margen_pct') or 0):.1f}%")
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
            cli = next((c for c in df.columns if "cliente" in unicodedata.normalize('NFD', c).lower()), None)
            val = next((c for c in df.columns if any(k in unicodedata.normalize('NFD', c).lower() for k in ["monto","neto","total","importe","facturacion","ingreso","venta","principal"])), None)
            if cli and val:
                vals = pd.to_numeric(df[val], errors="coerce")
                top = (df.assign(__v=vals).groupby(cli, dropna=False)["__v"].sum().sort_values(ascending=False).head(10))
                with c1d:
                    st.caption(f"Top-10 Clientes ({hoja})")
                    _barras_horizontal(top, cli, val, titulo=None)
                plotted1 = True
                break
        if not plotted1:  with c1d: st.info("No se encontró columna de cliente para Top-10.")

        # Estados
        plotted2 = False
        for hoja, df in data.items():
            est = next((c for c in df.columns if any(k in unicodedata.normalize('NFD', c).lower() for k in ["estado","resultado","situacion","situación","status"])), None)
            if est:
                dist = df[est].astype(str).str.strip().str.title().value_counts().head(8)
                if len(dist) >= 1:
                    with c2d:
                        st.caption(f"Distribución por Estado ({hoja})")
                        fig, ax = plt.subplots()
                        ax.pie(dist.values, labels=dist.index.astype(str), autopct="%1.0f%%", startangle=90)
                        ax.axis("equal"); st.pyplot(fig); st.download_button("⬇️ PNG", _export_fig(fig), "kpi_estado.png", "image/png")
                    plotted2 = True; break
        if not plotted2: with c2d: st.info("No se encontró columna de estado para distribución.")

        # Tendencia mensual
        ser_total = None
        for hoja, df in data.items():
            val = next((c for c in df.columns if any(k in unicodedata.normalize('NFD', c).lower() for k in ["monto","neto","total","importe","facturacion","ingreso","venta","principal"])), None)
            fec = next((c for c in df.columns if any(k in unicodedata.normalize('NFD', c).lower() for k in ["fecha","mes","emision","emisión"])), None)
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
                fig.autofmt_xdate(); st.pyplot(fig); st.download_button("⬇️ PNG", _export_fig(fig), "kpi_tendencia.png", "image/png")
        else:
            with c3d: st.info("No se detectaron fechas para tendencia mensual.")

# ----------- Consulta IA (estructurado + fallback) -----------
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
            # 1) intento estructurado (JSON)
            obj = ask_gpt_structured(prompt_analisis_general_JSON(analisis))
            if obj:
                render_structured(obj, data)
                st.session_state.historial.append({"pregunta":"Análisis general (JSON)","respuesta":json.dumps(obj, ensure_ascii=False)})
            else:
                # 2) fallback a texto + saneo
                raw = ask_gpt([
                    {"role":"system","content":"Eres un controller financiero. Responde en Markdown simple."},
                    {"role":"user","content":"Escribe un análisis con secciones estándar (resumen, diagnóstico, proyecciones, recomendaciones, riesgos, próximos pasos)."}
                ])
                texto, instr = split_text_and_viz(raw)
                texto = prettify_answer(texto)
                left, right = st.columns([0.58,0.42])
                with left: st.markdown(texto)
                with right:
                    if not render_viz_instructions(instr, data):
                        schema = _build_schema(data)
                        plan = plan_from_llm("Gráfico sugerido según KPIs", schema)
                        execute_plan(plan, data)
                st.session_state.historial.append({"pregunta":"Análisis general (fallback)","respuesta":texto})

        if c2b.button("Responder") and pregunta:
            schema = _build_schema(data)
            obj = ask_gpt_structured(prompt_consulta_libre_JSON(pregunta, schema))
            if obj:
                render_structured(obj, data)
                st.session_state.historial.append({"pregunta":pregunta,"respuesta":json.dumps(obj, ensure_ascii=False)})
            else:
                raw = ask_gpt([
                    {"role":"system","content":"Eres un controller financiero. Responde en Markdown simple."},
                    {"role":"user","content":pregunta}
                ])
                texto, instr = split_text_and_viz(raw)
                texto = prettify_answer(texto)
                left, right = st.columns([0.58,0.42])
                with left: st.markdown(texto)
                with right:
                    if not render_viz_instructions(instr, data):
                        plan = plan_from_llm(pregunta, schema); execute_plan(plan, data)
                st.session_state.historial.append({"pregunta":pregunta,"respuesta":texto})

# ----------- Historial -----------
elif st.session_state.menu_sel == "Historial":
    if st.session_state.historial:
        for i, h in enumerate(st.session_state.historial[-20:], 1):
            st.markdown(f"**Q{i}:** {h['pregunta']}")
            st.markdown(f"**A{i}:**")
            # para evitar renderizar JSON como Markdown extraño:
            try:
                parsed = json.loads(h["respuesta"])
                st.json(parsed)
            except Exception:
                st.markdown(prettify_answer(h["respuesta"]))
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
        if diag["quota_ok"] is True: st.success("Créditos/cuota: OK")
        elif diag["quota_ok"] is False: st.error("❌ Sin créditos/cuota.")
        else: st.info("No se pudo determinar el estado de la cuota.")
        if diag["usage_tokens"] is not None: st.caption(f"Tokens usados: {diag['usage_tokens']}")
        if diag["error"]: st.warning(f"Detalle: {diag['error']}")
