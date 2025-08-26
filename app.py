import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gspread
import io, json, re, unicodedata, os, inspect, hashlib
from html import escape, unescape
from typing import Dict, Any, List
from google.oauth2.service_account import Credentials
from openai import OpenAI
from streamlit.components.v1 import html as st_html
from analizador import analizar_datos_taller

# =======================
# CONFIG GENERAL
# =======================
APP_BUILD = "build-2025-08-26-verified-1"
st.set_page_config(layout="wide", page_title="Controller Financiero IA")

st.markdown("""
<style>
html, body, [data-testid="stMarkdownContainer"]{
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif !important;
  font-size: 15.5px !important; line-height: 1.55 !important;
}
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3{
  font-weight:700 !important; letter-spacing:.2px;
}
[data-testid="stMarkdownContainer"] em,[data-testid="stMarkdownContainer"] i{font-style:normal!important;}
[data-testid="stMarkdownContainer"] code{font-family:inherit!important;background:transparent!important;}
</style>
""", unsafe_allow_html=True)

# =======================
# ESTADO
# =======================
ss = st.session_state
ss.setdefault("historial", [])
ss.setdefault("data", None)
ss.setdefault("sheet_url", "")
ss.setdefault("max_cats_grafico", 18)
ss.setdefault("top_n_grafico", 12)
ss.setdefault("aliases", {})
ss.setdefault("menu_sel", "KPIs")

# =======================
# CARGA DE DATOS
# =======================
@st.cache_data(show_spinner=False, ttl=300)
def load_excel(file):
    return pd.read_excel(file, sheet_name=None)

@st.cache_data(show_spinner=False, ttl=300)
def load_gsheet(json_keyfile: str, sheet_url: str):
    creds = Credentials.from_service_account_info(
        json.loads(json_keyfile),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    return {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sheet.worksheets()}

# =======================
# OPENAI
# =======================
def _get_openai_client():
    api_key = (st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.error("⚠️ Falta `OPENAI_API_KEY` en Secrets.")
        return None
    api_key = str(api_key).strip().strip('"').strip("'")
    org = st.secrets.get("OPENAI_ORG") or os.getenv("OPENAI_ORG")
    base_url = st.secrets.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    kw = {"api_key": api_key}
    if org: kw["organization"] = str(org).strip()
    if base_url: kw["base_url"] = str(base_url).strip()
    try:
        return OpenAI(**kw)
    except Exception as e:
        st.error(f"No se pudo inicializar OpenAI: {e}")
        return None

def ask_gpt(messages) -> str:
    client = _get_openai_client()
    if client is None:
        return "⚠️ Error inicializando OpenAI."
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.15
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Fallo en la petición a OpenAI: {e}")
        return "⚠️ No pude completar la consulta a la IA."

def diagnosticar_openai():
    res = {
        "api_key_present": False, "organization_set": False, "base_url_set": False,
        "list_models_ok": False, "chat_ok": False, "quota_ok": None,
        "usage_tokens": None, "error": None
    }
    key = (st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY"))
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
        res["error"] = f"Error listando modelos: {e}"; return res
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":"Ping diagnóstico. Responde: OK."}],
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

# =======================
# NORMALIZACIÓN / TEXTO
# =======================
ALL_SPACES_RE = re.compile(r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]')
INVISIBLES_RE = re.compile(r'[\u200B\u200C\u200D\uFEFF\u2060\u00AD]')
TITLE_LINE_RE = re.compile(r'^(#{1,6}\s+[^\n]+)$', re.M)
CURRENCY_COMMA_RE = re.compile(r'(?<![\w$])(\d{1,3}(?:,\d{3})+)(?![\w%])')
CURRENCY_DOT_RE   = re.compile(r'(?<![\w$])(\d{1,3}(?:\.\d{3})+)(?![\w%])')
MILLONES_RE = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*millone?s\b')
MILES_RE    = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*mil\b')

def _to_clp(i: int) -> str:
    return f"${i:,}".replace(",", ".")

def prettify_answer(text: str) -> str:
    """Limpia texto y normaliza montos (CLP). Sin reglas que separen palabras."""
    if not text: return text
    t = unicodedata.normalize("NFKC", text)
    t = unescape(t)  # decodifica entidades si venían
    t = re.sub(r'<[^>]+>', '', t)
    t = INVISIBLES_RE.sub('', t)
    t = ALL_SPACES_RE.sub(' ', t)
    t = t.replace("•", "\n- ").replace("“","\"").replace("”","\"").replace("’","'")
    # quitar énfasis markdown
    t = re.sub(r'([*_`~]{1,3})(?=\S)(.+?)(?<=\S)\1', r'\2', t)
    t = TITLE_LINE_RE.sub(r'\1\n', t)
    t = re.sub(r'^[\s]*[-•]\s*', '- ', t, flags=re.M)
    t = re.sub(r'[ \t]+', ' ', t); t = re.sub(r'\n\s+', '\n', t)

    # errores frecuentes del modelo
    t = re.sub(r'(?i)\bmargende\b', 'margen de', t)
    t = re.sub(r'(?i)\bmientrasque\b', 'mientras que', t)

    # prefijos de moneda
    t = re.sub(r'(?i)(?:US|CLP|COP|S)\s*\$', '$', t)

    def _mill(m):
        try: return _to_clp(int(round(float(m.group(1).replace(",","."))*1_000_000)))
        except: return m.group(0)
    def _mil(m):
        try: return _to_clp(int(round(float(m.group(1).replace(",","."))*1_000)))
        except: return m.group(0)

    t = MILLONES_RE.sub(_mill, t)
    t = MILES_RE.sub(_mil, t)

    t = CURRENCY_COMMA_RE.sub(lambda m: _to_clp(int(m.group(1).replace(',',''))), t)
    t = CURRENCY_DOT_RE.sub(  lambda m: _to_clp(int(m.group(1).replace('.',''))), t)

    t = re.sub(r'\$\s*\$+', '$', t)      # $$ → $
    t = re.sub(r'\$\s+(?=\d)', '$', t)   # $ 1.000 → $1.000
    t = re.sub(r':(?=\S)', ': ', t)
    t = re.sub(r',(?=\S)', ', ', t)
    return t.strip()

def sanitize_text_for_html(s: str) -> str:
    """Neutraliza LaTeX y deja $ normal (no entidades)."""
    if not s: return ""
    t = unicodedata.normalize("NFKC", s)
    t = unescape(t)
    t = re.sub(r'[\u200B\u200C\u200D\uFEFF\u2060\u00AD]', '', t)
    t = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]', ' ', t)
    # neutraliza LaTeX
    t = re.sub(r'\\\((.*?)\\\)', r'\1', t, flags=re.S)
    t = re.sub(r'\\\[(.*?)\\\]', r'\1', t, flags=re.S)
    t = re.sub(r'\$\$(.*?)\$\$', r'\1', t, flags=re.S)
    t = re.sub(r'\$\s*\$+', '$', t)
    t = t.replace("•", "\n- ")
    t = re.sub(r'[ \t]+', ' ', t); t = re.sub(r'\n\s+','\n', t)
    return t.strip()

def md_to_safe_html(markdown_text: str) -> str:
    """Convierte '###' y '- ' a HTML plano escapado."""
    base = prettify_answer(markdown_text or "")
    t = sanitize_text_for_html(base)
    out, ul = [], False
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            if ul: out.append("</ul>"); ul = False
            continue
        if line.startswith("###"):
            if ul: out.append("</ul>"); ul = False
            out.append(f"<h3>{escape(line.lstrip('#').strip())}</h3>")
        elif line.startswith("- "):
            if not ul: out.append("<ul>"); ul = True
            out.append(f"<li>{escape(line[2:].strip())}</li>")
        else:
            if ul: out.append("</ul>"); ul = False
            out.append(f"<p>{escape(line)}</p>")
    if ul: out.append("</ul>")
    return "\n".join(out)

def render_ia_html_block(text: str, height: int = 560):
    """Render en iframe (sin Markdown/KaTeX de Streamlit)."""
    safe_html = md_to_safe_html(text or "")
    page = f"""<!doctype html><html><head><meta charset="utf-8">
    <style>
      :root {{ --font: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu,
               "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif; }}
      html,body {{ margin:0; padding:0; font: 15.5px/1.55 var(--font); }}
      h3 {{ margin:0 0 .5rem; font-weight:700; }}
      ul {{ margin:.25rem 0 .75rem 1.25rem; padding-left:1rem; }}
      li,p {{ margin:.25rem 0; letter-spacing:.2px; white-space:normal; }}
      em,i {{ font-style: normal !important; }}
      code {{ font-family: inherit !important; background: transparent !important; }}
    </style></head><body>{safe_html}</body></html>"""
    st_html(page, height=height, scrolling=True)

# =======================
# VIZ – UTILIDADES
# =======================
def _fmt_pesos(x, pos=None):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"${int(round(x)):,}".replace(",", ".")
    except Exception:
        return str(x)

def _export_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf.read()

def _norm(s: str) -> str:
    s = str(s).replace("\u00A0"," ").strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+',' ', s)
    return s.lower()

def find_col(df: pd.DataFrame, name: str):
    if not name: return None
    alias = ss.aliases.get(_norm(name))
    if alias and alias in df.columns: return alias
    tgt = _norm(name)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    candidates = [c for c in df.columns if tgt in _norm(c) or _norm(c).startswith(tgt[:4])]
    return candidates[0] if candidates else None

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
    ax.set_ylabel(col_valor)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    ax.tick_params(axis='x', rotation=45)
    for lbl in ax.get_xticklabels(): lbl.set_ha('right')
    for b in bars:
        y = b.get_height()
        if np.isfinite(y):
            ax.annotate(_fmt_pesos(y), (b.get_x()+b.get_width()/2, y), textcoords="offset points",
                        xytext=(0,3), ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

def _barras_horizontal(resumen, col_categoria, col_valor, titulo):
    fig, ax = plt.subplots()
    bars = ax.barh(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    ax.set_xlabel(col_valor)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    for b in bars:
        x = b.get_width()
        if np.isfinite(x):
            ax.annotate(_fmt_pesos(x), (x, b.get_y()+b.get_height()/2), textcoords="offset points",
                        xytext=(5,0), ha='left', va='center', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

# === NUEVA FUNCIÓN (evita colisiones con versiones viejas) ===
def mostrar_grafico_barras_v3(df, col_categoria, col_valor, titulo=None, top_n=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals)
                 .groupby(col_categoria, dropna=False)["__v"]
                 .sum()
                 .sort_values(ascending=False))

    try:
        if len(resumen) >= 8:
            top_val = float(resumen.iloc[0])
            med = float(np.median(resumen.values))
            if med > 0 and top_val / med >= 3.0:
                st.info("Distribución desbalanceada: muestro tabla para mejor lectura.")
                mostrar_tabla(df, col_categoria, col_valor, titulo)
                return
    except Exception:
        pass

    if top_n is None:
        top_n = ss.get("top_n_grafico", 12)

    recorte = False
    if len(resumen) > top_n:
        resumen = resumen.head(top_n)
        recorte = True

    # Cálculo robusto de longitud media SIN .str sobre Index
    labels = [str(x) for x in resumen.index.tolist()]
    avg_len = float(np.mean([len(s) for s in labels])) if labels else 0.0

    try:
        if avg_len > 10:
            _barras_horizontal(resumen, col_categoria, col_valor, titulo)
        else:
            _barras_vertical(resumen, col_categoria, col_valor, titulo)
    except Exception as e:
        st.error(f"No pude generar el gráfico: {e}. Muestro tabla como respaldo.")
        mostrar_tabla(df, col_categoria, col_valor, titulo)
        return

    if recorte:
        st.caption(f"Mostrando Top-{top_n}. Usa tabla para el detalle completo.")

def mostrar_grafico_torta(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))
    fig, ax = plt.subplots()
    ax.pie(resumen.values, labels=[str(x) for x in resumen.index], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    st.pyplot(fig)
    st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png")

# =======================
# PARSER 'viz:' EN RESPUESTA
# =======================
VIZ_PATT = re.compile(r'(?:^|\n)\s*(?:viz\s*:\s*([^\n\r]+))', re.IGNORECASE)

def split_text_and_viz(respuesta_texto: str):
    text = respuesta_texto
    instr = []
    for m in VIZ_PATT.finditer(respuesta_texto):
        body = m.group(1).strip().strip("`").lstrip("-*• ").strip()
        parts = [p.strip(" `*-•").strip() for p in body.split("|")]
        if len(parts) >= 3:
            instr.append(("viz", parts))
        text = text.replace(m.group(0), "")
    return text.strip(), instr

def _safe_plot(plot_fn, hoja, df, cat_raw, val_raw, titulo):
    cat = find_col(df, cat_raw); val = find_col(df, val_raw)
    if not cat or not val:
        st.warning(f"❗ No se pudo generar en '{hoja}'. Revisar columnas: '{cat_raw}' y '{val_raw}'.")
        return
    try:
        plot_fn(df, cat, val, titulo)
        ss.aliases[_norm(cat_raw)] = cat
        ss.aliases[_norm(val_raw)] = val
    except Exception as e:
        st.error(f"Error generando visualización en '{hoja}': {e}")

def render_viz_instructions(instr_list, data_dict):
    if not instr_list: return False
    for item in instr_list:
        _tipo, parts = item
        tipo = parts[0].lower()
        cat_raw, val_raw = parts[1], parts[2]
        titulo = parts[3] if len(parts) >= 4 else None
        for hoja, df in data_dict.items():
            if find_col(df, cat_raw) and find_col(df, val_raw):
                try:
                    if tipo == "barras":
                        _safe_plot(mostrar_grafico_barras_v3, hoja, df, cat_raw, val_raw, titulo); return True
                    if tipo == "torta":
                        _safe_plot(mostrar_grafico_torta, hoja, df, cat_raw, val_raw, titulo); return True
                    mostrar_tabla(df, find_col(df, cat_raw), find_col(df, val_raw), titulo or f"Tabla ({hoja})"); return True
                except Exception as e:
                    st.error(f"Error generando visualización en '{hoja}': {e}")
                    return False
    return False

# =======================
# PLANNER (visual) y EXEC
# =======================
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
    if action not in ("table", "chart"):
        return False

    sheet = plan.get("sheet") or ""
    cat = plan.get("category_col") or ""
    val = plan.get("value_col") or ""
    chart = (plan.get("chart") or "auto").lower()
    title = plan.get("title") or None

    hojas = [sheet] if sheet in data else list(data.keys())

    for h in hojas:
        df = data[h]
        if df is None or df.empty:
            continue

        cat_real = find_col(df, cat) if cat else None
        val_real = find_col(df, val) if val else None
        if not (cat_real and val_real):
            continue

        try:
            if action == "table":
                mostrar_tabla(df, cat_real, val_real, title)
                return True

            if chart in ("auto", "barras"):
                mostrar_grafico_barras_v3(df, cat_real, val_real, title)
                return True

            if chart == "torta":
                mostrar_grafico_torta(df, cat_real, val_real, title)
                return True
        except Exception as e:
            st.error(f"Error generando visualización en '{h}': {e}")
            return False

    return False

# =======================
# PLANNER DE CÓMPUTO (VERIFICADO)
# =======================
def plan_compute_from_llm(pregunta: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    system = ("Eres un planificador de CÓMPUTO financiero. Devuelve SOLO JSON. "
              "NUNCA inventes columnas. Si la métrica es monetaria, usa op='sum' por defecto.")
    prompt = f"""
Devuelve SOLO este JSON (sin texto adicional):
{{
  "sheet": "<nombre_hoja_o_vacio_para_auto>",
  "value_col": "<columna_de_valor>",
  "category_col": "<columna_categoria_o_vacio_si_no_aplica>",
  "op": "sum|avg|count|max|min",
  "filters": [{{"col":"<col>","op":"eq|contains|gte|lte","val":"<valor>"}}]
}}

Reglas:
- Si el usuario pide "por X", llena category_col con X y AGREGA TODOS los X (no solo el primero).
- Si no hay "por", deja category_col vacío y calcula el TOTAL.
- No inventes nombres: usa EXACTOS del esquema (insensible a mayúsculas).
- Si dudas de la hoja, deja "sheet": "" para que el sistema decida.

ESQUEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

PREGUNTA:
{pregunta}
"""
    raw = ask_gpt([{"role":"system","content":system},
                   {"role":"user","content":prompt}]).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _apply_filters(df: pd.DataFrame, filters: List[Dict[str,str]]) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for f in filters:
        col = find_col(out, f.get("col",""))
        if not col:
            continue
        op  = (f.get("op","eq") or "eq").lower()
        val = str(f.get("val",""))
        if op == "eq":
            out = out[out[col].astype(str).str.lower() == val.lower()]
        elif op == "contains":
            out = out[out[col].astype(str).str.contains(val, case=False, na=False)]
        elif op == "gte":
            out = out[pd.to_numeric(out[col], errors="coerce") >= pd.to_numeric(val, errors="coerce")]
        elif op == "lte":
            out = out[pd.to_numeric(out[col], errors="coerce") <= pd.to_numeric(val, errors="coerce")]
    return out

def execute_compute(plan: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve:
    {
      "ok": True/False, "msg": "...",
      "sheet": str, "value_col": str, "category_col": str, "op": str,
      "rows": int, "total": float,
      "by_category": [{"categoria": "...", "valor": float}, ...],  # si category_col != ""
      "df_result": DataFrame
    }
    """
    if not plan:
        return {"ok": False, "msg": "Plan vacío."}

    sheet = plan.get("sheet","") or ""
    val_raw = plan.get("value_col","") or ""
    cat_raw = plan.get("category_col","") or ""
    op = (plan.get("op","sum") or "sum").lower()
    filters = plan.get("filters", [])

    hojas = [sheet] if sheet in data else list(data.keys())
    for h in hojas:
        df = data[h]
        if df is None or df.empty:
            continue
        val_col = find_col(df, val_raw)
        if not val_col:
            continue
        dff = _apply_filters(df, filters)
        vals = pd.to_numeric(dff[val_col], errors="coerce")

        if cat_raw:
            cat_col = find_col(dff, cat_raw)
            if not cat_col:
                continue
            g = dff.assign(__v=vals).groupby(cat_col, dropna=False)["__v"]
            if op == "avg":
                s = g.mean()
            elif op == "count":
                s = g.count()
            elif op == "max":
                s = g.max()
            elif op == "min":
                s = g.min()
            else:
                s = g.sum()
            s = s.sort_values(ascending=False)
            total = float(pd.to_numeric(s, errors="coerce").sum())
            df_res = s.reset_index()
            df_res.columns = [str(cat_col), str(val_col)]
            return {
                "ok": True, "msg": "", "sheet": h,
                "value_col": val_col, "category_col": cat_col, "op": op,
                "rows": int(len(dff)), "total": total,
                "by_category": [{"categoria": str(k), "valor": float(v) if pd.notna(v) else 0.0}
                                for k, v in s.items()],
                "df_result": df_res
            }
        else:
            # TOTAL simple
            if op == "avg":
                total = float(vals.mean())
            elif op == "count":
                total = int(vals.count())
            elif op == "max":
                total = float(vals.max())
            elif op == "min":
                total = float(vals.min())
            else:
                total = float(vals.sum())
            df_res = pd.DataFrame({str(val_col): [total], "TOTAL": ["TOTAL"]})[[ "TOTAL", str(val_col)]]
            return {
                "ok": True, "msg":"", "sheet": h,
                "value_col": val_col, "category_col": "", "op": op,
                "rows": int(len(dff)), "total": total,
                "by_category": [],
                "df_result": df_res
            }

    return {"ok": False, "msg": "No se encontraron columnas compatibles en las hojas."}

# ===== Resumen verificado + secciones cualitativas (sin cifras nuevas) =====
def _fmt_list_top(by_category, top=10):
    out = []
    for i, item in enumerate(by_category[:top], 1):
        cat = str(item["categoria"])
        val = _fmt_pesos(item["valor"])
        out.append(f"- {cat}: {val}")
    if len(by_category) > top:
        out.append(f"- (… {len(by_category)-top} más)")
    return "\n".join(out)

def build_verified_summary(facts: dict) -> str:
    val = facts.get("value_col","valor")
    cat = facts.get("category_col","")
    total = _fmt_pesos(facts.get("total",0))
    rows = facts.get("rows",0)

    encabezado = "### Resumen ejecutivo\n"
    if cat:
        bullets = [
            f"- {val.title()} total por **{cat}**: {total} (sobre {rows} filas).",
            "- Detalle por categoría:",
            _fmt_list_top(facts.get("by_category", []), top=10)
        ]
    else:
        bullets = [
            f"- {val.title()} total: {total} (sobre {rows} filas)."
        ]
    return encabezado + "\n".join(bullets)

def qualitative_sections(facts: dict, pregunta: str) -> str:
    system = ("Eres un analista financiero. Redacta secciones claras SIN introducir cifras nuevas: "
              "si necesitas referirte a montos, usa expresiones cualitativas (alto/bajo/estable).")
    facts_min = {
        "sheet": facts.get("sheet"),
        "value_col": facts.get("value_col"),
        "category_col": facts.get("category_col"),
    }
    prompt = f"""
Contexto (no inventes números):
{json.dumps(facts_min, ensure_ascii=False)}

Pregunta del usuario:
{pregunta}

Escribe exactamente estas secciones, sin cifras:
### Diagnóstico
- …

### Estimaciones y proyecciones
- …

### Recomendaciones de gestión
- …

### Riesgos y alertas
- …

### Próximos pasos
- …
"""
    txt = ask_gpt([{"role":"system","content":system},{"role":"user","content":prompt}])
    return txt or ""

# =======================
# PROMPTS IA (análisis general / legacy)
# =======================
def make_system_prompt():
    return ("Eres un Controller Financiero senior para un taller de desabolladura y pintura. "
            "Responde SIEMPRE con estilo ejecutivo + analítico y basándote EXCLUSIVAMENTE en la planilla.")

ANALYSIS_FORMAT = """
Escribe SIEMPRE en este formato (usa '###' y bullets '- '):

### Resumen ejecutivo
- 3 a 5 puntos clave con cifras y contexto.

### Diagnóstico
- Qué está bien / mal y por qué.

### Estimaciones y proyecciones
- Proyección 3–6 meses (escenarios: optimista/base/conservador) con supuestos.

### Recomendaciones de gestión
- 5–8 acciones priorizadas (impacto y dificultad).

### Riesgos y alertas
- 3–5 riesgos y mitigaciones.

### Próximos pasos
- Dueños, plazos y métrica objetivo.

Al final, si aplica, añade UNA instrucción:
viz: barras|<categoria>|<valor>|<título opcional>
viz: torta|<categoria>|<valor>|<título opcional>
viz: tabla|<categoria>|<valor>|<título opcional>
"""

def prompt_analisis_general(analisis_dict: dict) -> list:
    return [
        {"role":"system","content": make_system_prompt()},
        {"role":"user","content": f"""
Con base en los siguientes KPIs calculados reales, realiza un ANÁLISIS PROFESIONAL siguiendo el formato obligatorio.
No inventes datos fuera de lo entregado; si necesitas supuestos, decláralos conservadoramente.

KPIs:
{json.dumps(analisis_dict, ensure_ascii=False, indent=2)}

{ANALYSIS_FORMAT}
"""}
    ]

def prompt_consulta_libre(pregunta: str, schema: dict) -> list:
    historial_msgs = []
    for h in ss.historial[-6:]:
        historial_msgs += [{"role":"user","content":h["pregunta"]},
                           {"role":"assistant","content":h["respuesta"]}]
    return [
        {"role":"system","content": make_system_prompt()},
        *historial_msgs,
        {"role":"user","content": f"""
Contesta usando el esquema de datos; incluye SIEMPRE el formato completo y, si aplica, UNA instrucción de visualización.
Pregunta: {pregunta}

Esquema (hojas, columnas y ejemplos):
{json.dumps(schema, ensure_ascii=False, indent=2)}

{ANALYSIS_FORMAT}
"""}
    ]

# =======================
# UI
# =======================
st.title("🤖 Controller Financiero IA")

with st.sidebar:
    st.markdown("### Menú")
    ss.menu_sel = st.radio(
        "Secciones",
        ["Datos","Vista previa","KPIs","Consulta IA","Historial","Diagnóstico IA"],
        index=["Datos","Vista previa","KPIs","Consulta IA","Historial","Diagnóstico IA"].index(ss.menu_sel),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Preferencias")
    ss.max_cats_grafico = st.number_input("Máx. categorías para graficar", 6, 200, ss.max_cats_grafico)
    ss.top_n_grafico    = st.number_input("Top-N por defecto (barras)", 5, 100, ss.top_n_grafico)

    # ===== Panel de diagnóstico de código =====
    with st.expander("🔧 Diagnóstico del código en ejecución"):
        st.caption(f"Build: **{APP_BUILD}**")
        st.caption(f"Ruta archivo: `{__file__}`")
        try:
            with open(__file__, "r", encoding="utf-8") as f:
                src = f.read()
            st.caption(f"Definiciones 'mostrar_grafico_barras': {src.count('def mostrar_grafico_barras(')}")
            st.caption(f"¿Aparece 'labels.str.len('?: {'sí' if 'labels.str.len(' in src else 'no'}")
            h = hashlib.sha256(inspect.getsource(mostrar_grafico_barras_v3).encode("utf-8")).hexdigest()[:16]
            st.caption(f"Hash mostrar_grafico_barras_v3: `{h}`")
        except Exception as e:
            st.caption(f"No pude inspeccionar el archivo: {e}")

# ---- Datos
if ss.menu_sel == "Datos":
    st.markdown("### 📁 Datos")
    fuente = st.radio("Fuente", ["Excel","Google Sheets"], key="k_fuente")
    if fuente == "Excel":
        file = st.file_uploader("Sube un Excel", type=["xlsx","xls"])
        if file:
            ss.data = load_excel(file)
            st.success("Excel cargado.")
    else:
        with st.form(key="form_gsheet"):
            url = st.text_input("URL de Google Sheet", value=ss.sheet_url)
            conectar = st.form_submit_button("Conectar")
        if conectar and url:
            try:
                ss.data = load_gsheet(st.secrets["GOOGLE_CREDENTIALS"], url)
                ss.sheet_url = url
                st.success("Google Sheet conectado.")
            except Exception as e:
                st.error(f"Error conectando Google Sheet: {e}")

# ---- Vista previa
elif ss.menu_sel == "Vista previa":
    data = ss.data
    if not data:
        st.info("Carga datos en la sección **Datos**.")
    else:
        st.markdown("### 📄 Hojas")
        for name, df in data.items():
            st.markdown(f"#### 📘 {name} • filas: {len(df)}")
            st.dataframe(df.head(10), use_container_width=True)

# ---- KPIs
elif ss.menu_sel == "KPIs":
    data = ss.data
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

# ---- Consulta IA
elif ss.menu_sel == "Consulta IA":
    data = ss.data
    if not data:
        st.info("Carga datos en la sección **Datos**.")
    else:
        st.markdown("### 🤖 Consulta")
        pregunta = st.text_area("Pregunta")
        c1b, c2b = st.columns(2)
        left, right = st.columns([0.58, 0.42])

        # ---- Botón: Análisis general (legacy con visual planner)
        if c1b.button("📊 Análisis General Automático"):
            analisis = analizar_datos_taller(data, "")
            raw = ask_gpt(prompt_analisis_general(analisis))
            texto, instr = split_text_and_viz(raw)
            with left:
                render_ia_html_block(texto, height=620)
            with right:
                ok = False
                try:
                    ok = render_viz_instructions(instr, data)
                except Exception as e:
                    st.error(f"Error en instrucción de visualización: {e}")
                if not ok:
                    try:
                        schema = _build_schema(data)
                        plan = plan_from_llm("Sugerir mejor visual", schema)
                        execute_plan(plan, data)
                    except Exception as e:
                        st.error(f"Error ejecutando plan: {e}")
            ss.historial.append({"pregunta":"Análisis general","respuesta":texto})

        # ---- Botón: Responder (MODO VERIFICADO)
        if c2b.button("Responder") and pregunta:
            schema = _build_schema(data)
            # 1) Plan de cómputo (IA decide columnas/operación sin calcular)
            plan_c = plan_compute_from_llm(pregunta, schema)
            # 2) Cálculo con pandas (nuestros números)
            facts = execute_compute(plan_c, data)

            if not facts.get("ok"):
                with left:
                    st.error(f"No pude calcular con precisión: {facts.get('msg')}. Uso ruta clásica.")
                # Fallback clásico (evitar quedarse sin respuesta)
                raw = ask_gpt(prompt_consulta_libre(pregunta, schema))
                texto, instr = split_text_and_viz(raw)
                with left:
                    render_ia_html_block(texto, height=620)
                with right:
                    ok = False
                    try:
                        ok = render_viz_instructions(instr, data)
                    except Exception as e:
                        st.error(f"Error en instrucción de visualización: {e}")
                    if not ok:
                        try:
                            plan = plan_from_llm(pregunta, schema)
                            execute_plan(plan, data)
                        except Exception as e:
                            st.error(f"Error ejecutando plan: {e}")
                ss.historial.append({"pregunta":pregunta,"respuesta":texto})
            else:
                # 3) Texto verificado: resumen con cifras + cualitativo sin cifras
                texto = build_verified_summary(facts) + "\n\n" + qualitative_sections(facts, pregunta)
                with left:
                    render_ia_html_block(texto, height=620)

                # 4) Visualización basada en nuestro DataFrame (coherente con texto)
                with right:
                    df_res = facts["df_result"]
                    if facts.get("category_col"):
                        try:
                            mostrar_grafico_barras_v3(
                                df_res.rename(columns={facts["category_col"]: "CATEGORIA",
                                                       facts["value_col"]: "VALOR"}),
                                "CATEGORIA", "VALOR",
                                f"{facts['op'].upper()} de {facts['value_col']} por {facts['category_col']}"
                            )
                        except Exception as e:
                            st.error(f"Error graficando: {e}")
                            st.dataframe(df_res, use_container_width=True)
                    else:
                        st.metric(f"{facts['op'].upper()} de {facts['value_col']}", _fmt_pesos(facts["total"]))
                        st.dataframe(df_res, use_container_width=True)

                    st.caption(f"Hoja: {facts['sheet']} • Filas consideradas: {facts['rows']} • TOTAL: {_fmt_pesos(facts['total'])}")

                ss.historial.append({"pregunta":pregunta,"respuesta":texto})

# ---- Historial
elif ss.menu_sel == "Historial":
    if ss.historial:
        for i, h in enumerate(ss.historial[-20:], 1):
            st.markdown(f"**Q{i}:** {h['pregunta']}")
            st.markdown(f"**A{i}:**")
            try:
                parsed = json.loads(h["respuesta"])
                st.json(parsed)
            except Exception:
                render_ia_html_block(h["respuesta"], height=520)
    else:
        st.info("Aún no hay historial en esta sesión.")

# ---- Diagnóstico IA
elif ss.menu_sel == "Diagnóstico IA":
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

