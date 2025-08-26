import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gspread, io, json, re, unicodedata, os
from html import escape
from typing import Dict, Any, List
from google.oauth2.service_account import Credentials
from openai import OpenAI
from streamlit.components.v1 import html as st_html
from analizador import analizar_datos_taller

# =============== Config ===============
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

# =============== Estado ===============
ss = st.session_state
ss.setdefault("historial", [])
ss.setdefault("data", None)
ss.setdefault("sheet_url", "")
ss.setdefault("max_cats_grafico", 18)
ss.setdefault("top_n_grafico", 12)
ss.setdefault("aliases", {})
ss.setdefault("menu_sel", "KPIs")
ss.setdefault("debug_render", False)

# =============== Carga de datos ===============
@st.cache_data(show_spinner=False, ttl=300)
def load_excel(file): return pd.read_excel(file, sheet_name=None)

@st.cache_data(show_spinner=False, ttl=300)
def load_gsheet(json_keyfile: str, sheet_url: str):
    creds = Credentials.from_service_account_info(
        json.loads(json_keyfile),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url)
    return {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sheet.worksheets()}

# =============== OpenAI ===============
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
    if client is None: return "⚠️ Error inicializando OpenAI."
    try:
        r = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.15)
        return r.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Fallo en la petición a OpenAI: {e}")
        return "⚠️ No pude completar la consulta a la IA."

# =============== Normalización de texto y montos ===============
LETTER = r"A-Za-zÁÉÍÓÚÜÑáéíóúüñ"
ALL_SPACES_RE = re.compile(r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]')
INVISIBLES_RE = re.compile(r'[\u200B\u200C\u200D\uFEFF\u2060\u00AD]')
TITLE_LINE_RE = re.compile(r'^(#{1,6}\s+[^\n]+)$', re.M)
CURRENCY_COMMA_RE = re.compile(r'(?<![\w$])(\d{1,3}(?:,\d{3})+)(?![\w%])')
CURRENCY_DOT_RE   = re.compile(r'(?<![\w$])(\d{1,3}(?:\.\d{3})+)(?![\w%])')
MILLONES_RE = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*millone?s\b')
MILES_RE    = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*mil\b')

def _to_clp(i: int) -> str: return f"${i:,}".replace(",", ".")

def prettify_answer(text: str) -> str:
    """Limpieza general + normalización financiera (CLP). Sin inyección de conectores."""
    if not text: return text
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r'<[^>]+>', '', t)                    # quita HTML
    t = INVISIBLES_RE.sub('', t)                     # invisibles
    t = ALL_SPACES_RE.sub(' ', t)                    # espacios raros → espacio
    t = t.replace("•", "\n- ").replace("“","\"").replace("”","\"").replace("’","'")
    # quita marcadores de énfasis sin tocar espacios
    t = re.sub(r'([*_`~]{1,3})(?=\S)(.+?)(?<=\S)\1', r'\2', t)
    t = TITLE_LINE_RE.sub(r'\1\n', t)
    t = re.sub(r'^[\s]*[-•]\s*', '- ', t, flags=re.M)
    t = re.sub(r'[ \t]+', ' ', t); t = re.sub(r'\n\s+', '\n', t)

    # correcciones puntuales de errores frecuentes
    t = re.sub(r'(?i)\bmargende\b', 'margen de', t)
    t = re.sub(r'(?i)\bmientrasque\b', 'mientras que', t)

    # normaliza prefijos de moneda y cantidades en palabras
    t = re.sub(r'(?i)(?:US|CLP|COP|S)\s*\$', '$', t)
    def _mill(m):
        try: return _to_clp(int(round(float(m.group(1).replace(",","."))*1_000_000)))
        except: return m.group(0)
    def _mil(m):
        try: return _to_clp(int(round(float(m.group(1).replace(",","."))*1_000)))
        except: return m.group(0)
    t = MILLONES_RE.sub(_mill, t)
    t = MILES_RE.sub(_mil, t)

    # miles con coma o punto
    t = CURRENCY_COMMA_RE.sub(lambda m: _to_clp(int(m.group(1).replace(',',''))), t)
    t = CURRENCY_DOT_RE.sub(  lambda m: _to_clp(int(m.group(1).replace('.',''))), t)

    # $$ → $, $ 1.000 → $1.000
    t = re.sub(r'\$\s*\$+', '$', t)
    t = re.sub(r'\$\s+(?=\d)', '$', t)

    # espacios básicos
    t = re.sub(r':(?=\S)', ': ', t)
    t = re.sub(r',(?=\S)', ', ', t)
    return t.strip()

def sanitize_text_for_html(s: str) -> str:
    """Neutraliza delimitadores LaTeX. No convierte $ a entidades."""
    if not s: return ""
    t = unicodedata.normalize("NFKC", s)
    t = re.sub(r'[\u200B\u200C\u200D\uFEFF\u2060\u00AD]', '', t)
    t = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]', ' ', t)
    # neutraliza LaTeX
    t = re.sub(r'\\\((.*?)\\\)', r'\1', t, flags=re.S)
    t = re.sub(r'\\\[(.*?)\\\]', r'\1', t, flags=re.S)
    t = re.sub(r'\$\$(.*?)\$\$', r'\1', t, flags=re.S)
    # colapsa $$ -> $
    t = re.sub(r'\$\s*\$+', '$', t)
    # bullets y espacios
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
            if ul: out.append("</ul>"); ul = False; continue
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

def render_ia_html_block(text: str, height: int = 560, debug_label: str = ""):
    safe_html = md_to_safe_html(text or "")
    if ss.debug_render:
        with st.expander(f"🔧 Debug {debug_label} – RAW modelo"):
            st.code((text or "")[:4000])
        with st.expander(f"🔧 Debug {debug_label} – prettify_answer()"):
            st.code(prettify_answer(text or "")[:4000])
        with st.expander(f"🔧 Debug {debug_label} – HTML final"):
            st.code(safe_html[:4000], language="html")

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

# =============== Utilidades VIZ ===============
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

def mostrar_grafico_barras(df, col_categoria, col_valor, titulo=None, top_n=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))
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

    if top_n is None: top_n = ss.get("top_n_grafico", 12)
    recorte = False
    if len(resumen) > top_n:
        resumen = resumen.head(top_n); recorte = True

    # >>> FIX: usa Series para .str.len().mean()
    labels = pd.Series(resumen.index.astype(str))
    avg_len = labels.str.len().mean()

    if avg_len > 10:
        _barras_horizontal(resumen, col_categoria, col_valor, titulo)
    else:
        _barras_vertical(resumen, col_categoria, col_valor, titulo)

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

# =============== Prompts ===============
def make_system_prompt():
    return ("Eres un Controller Financiero senior para un taller. "
            "Responde SIEMPRE con estilo ejecutivo + analítico y 100% basado en la planilla.")

ANALYSIS_FORMAT = """
Escribe SIEMPRE en este formato (usa '###' y bullets '- '):

### Resumen ejecutivo
- 3 a 5 puntos con cifras y contexto.
### Diagnóstico
- Qué está bien / mal y por qué.
### Estimaciones y proyecciones
- 3–6 meses, escenarios (optimista/base/conservador) con supuestos.
### Recomendaciones de gestión
- 5–8 acciones priorizadas (impacto/dificultad).
### Riesgos y alertas
- 3–5 riesgos y mitigaciones.
### Próximos pasos
- Dueños, plazos y métrica.
"""

def prompt_analisis_general(kpis: dict) -> list:
    return [{"role":"system","content": make_system_prompt()},
            {"role":"user","content": f"KPIs:\n{json.dumps(kpis, ensure_ascii=False, indent=2)}\n{ANALYSIS_FORMAT}"}]

def prompt_consulta_libre(q: str, schema: dict) -> list:
    hist = []
    for h in ss.historial[-6:]:
        hist += [{"role":"user","content":h["pregunta"]},{"role":"assistant","content":h["respuesta"]}]
    return [{"role":"system","content": make_system_prompt()}, *hist,
            {"role":"user","content": f"Pregunta:\n{q}\nEsquema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n{ANALYSIS_FORMAT}"}]

# =============== UI ===============
st.title("🤖 Controller Financiero IA")
with st.sidebar:
    st.markdown("### Menú")
    ss.menu_sel = st.radio("Secciones",
        ["Datos","Vista previa","KPIs","Consulta IA","Historial","Diagnóstico IA"],
        index=["Datos","Vista previa","KPIs","Consulta IA","Historial","Diagnóstico IA"].index(ss.menu_sel),
        label_visibility="collapsed")
    st.markdown("---")
    ss.debug_render = st.toggle("🔧 Depurar render de IA", value=ss.debug_render)
    st.markdown("### Preferencias")
    ss.max_cats_grafico = st.number_input("Máx. categorías para graficar", 6, 200, ss.max_cats_grafico)
    ss.top_n_grafico   = st.number_input("Top-N por defecto (barras)", 5, 100, ss.top_n_grafico)

# --- Datos
if ss.menu_sel == "Datos":
    st.markdown("### 📁 Datos")
    fuente = st.radio("Fuente", ["Excel","Google Sheets"], key="k_fuente")
    if fuente == "Excel":
        file = st.file_uploader("Sube un Excel", type=["xlsx","xls"])
        if file: ss.data = load_excel(file); st.success("Excel cargado.")
    else:
        with st.form(key="form_gsheet"):
            url = st.text_input("URL de Google Sheet", value=ss.sheet_url)
            if st.form_submit_button("Conectar"):
                try:
                    ss.data = load_gsheet(st.secrets["GOOGLE_CREDENTIALS"], url); ss.sheet_url = url
                    st.success("Conectado.")
                except Exception as e:
                    st.error(f"Error conectando Google Sheet: {e}")

# --- Vista previa
elif ss.menu_sel == "Vista previa":
    if not ss.data: st.info("Carga datos en **Datos**.")
    else:
        for name, df in ss.data.items():
            st.markdown(f"#### 📘 {name} • filas: {len(df)}")
            st.dataframe(df.head(10), use_container_width=True)

# --- KPIs
elif ss.menu_sel == "KPIs":
    data = ss.data
    if not data: st.info("Carga datos en **Datos**.")
    else:
        k = analizar_datos_taller(data, "")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Ingresos ($)", f"{int(round(k['ingresos'])):,}".replace(",", "."))
        c2.metric("Costos ($)",   f"{int(round(k['costos'])):,}".replace(",", "."))
        c3.metric("Margen ($)",   f"{int(round(k['margen'])):,}".replace(",", "."))
        c4.metric("Margen %",     f"{(k.get('margen_pct') or 0):.1f}%")
        st.caption(f"Servicios: {k.get('servicios',0)} | Ticket promedio: {('$'+format(int(round(k.get('ticket_promedio',0))),',').replace(',','.')) if k.get('ticket_promedio') else '—'}")

# --- Consulta IA
elif ss.menu_sel == "Consulta IA":
    data = ss.data
    if not data: st.info("Carga datos en **Datos**.")
    else:
        st.markdown("### 🤖 Consulta")
        pregunta = st.text_area("Pregunta")
        b1, b2 = st.columns(2)
        left, right = st.columns([0.58, 0.42])

        if b1.button("📊 Análisis General Automático"):
            kpis = analizar_datos_taller(data, "")
            raw = ask_gpt(prompt_analisis_general(kpis))
            with left:  render_ia_html_block(raw, height=620, debug_label="Análisis general")
            with right: st.info("Puedes pedir un gráfico específico en la pregunta.")
            ss.historial.append({"pregunta":"Análisis general","respuesta": raw})

        if b2.button("Responder") and pregunta:
            schema = {h: {"columns": list(df.columns)} for h,df in data.items()}
            raw = ask_gpt(prompt_consulta_libre(pregunta, schema))
            with left:  render_ia_html_block(raw, height=620, debug_label="Consulta libre")
            with right: st.info("Si la respuesta incluye 'viz: ...', genera el gráfico correspondiente.")
            ss.historial.append({"pregunta":pregunta,"respuesta": raw})

# --- Historial
elif ss.menu_sel == "Historial":
    if not ss.historial: st.info("Aún no hay historial.")
    else:
        for i,h in enumerate(ss.historial[-20:],1):
            st.markdown(f"**Q{i}:** {h['pregunta']}")
            st.markdown(f"**A{i}:**")
            try: st.json(json.loads(h["respuesta"]))
            except Exception: render_ia_html_block(h["respuesta"], height=520, debug_label=f"Historial {i}")

# --- Diagnóstico IA
elif ss.menu_sel == "Diagnóstico IA":
    st.markdown("### 🔎 Diagnóstico de la IA (OpenAI)")
    if st.button("Diagnosticar IA"):
        client = _get_openai_client()
        st.write("Cliente inicializado:", "✅" if client else "❌")
        if client:
            try:
                _ = client.models.list(); st.write("Listar modelos:", "✅")
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":"Ping diagnóstico. Di: OK"}],
                    temperature=0, max_tokens=4
                )
                st.write("Prueba de chat:", "✅", "| tokens:", getattr(getattr(r, "usage", None), "total_tokens", None))
            except Exception as e:
                st.error(f"Error: {e}")

