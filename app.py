# app.py
# Controller Financiero IA — build: 2025-08-27 focus-v7b

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gspread
import io, json, re, unicodedata, os, hashlib
from html import unescape, escape
from typing import Dict, Any, List, Tuple, Optional
from google.oauth2.service_account import Credentials
from openai import OpenAI
from streamlit.components.v1 import html as st_html

from analizador import analizar_datos_taller

APP_BUILD = "build-2025-08-27-focus-v7b"
st.set_page_config(layout="wide", page_title="Controller Financiero IA")


# =========================
# 🔐 Login + Logo (added)
# =========================
# Lee secretos (o usa defaults seguros)
def _get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

# Credenciales de acceso (puedes setear en .streamlit/secrets.toml)
_USER_DEFAULT = "admin"
_PASS_DEFAULT = "1234"
USER = _get_secret("USER", _USER_DEFAULT) or _USER_DEFAULT
PASSWORD = _get_secret("PASSWORD", _PASS_DEFAULT) or _PASS_DEFAULT

# Logo configurable
LOGO_PATH = _get_secret("LOGO_PATH", "Fenix_isotipo.png") or "Fenix_isotipo.png"
try:
    LOGO_SIZE_LOGIN = int(_get_secret("LOGO_SIZE_LOGIN", 56) or 56)
except Exception:
    LOGO_SIZE_LOGIN = 56
try:
    LOGO_SIZE_APP = int(_get_secret("LOGO_SIZE_APP", 42) or 42)
except Exception:
    LOGO_SIZE_APP = 42

# Estado de autenticación
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def render_logo_topright(size_px: int):
    """Muestra el logo alineado arriba a la derecha con ancho fijo (px)."""
    try:
        spacer, col_logo = st.columns([1, 0.14], vertical_alignment="center")
        with col_logo:
            st.image(LOGO_PATH, width=int(size_px))
    except Exception:
        # Si falla la carga, no interrumpimos la app
        pass

def _logout():
    st.session_state.authenticated = False
    # Limpia cachés/datos si aplica
    for k in ["data", "gsheet_url", "excel_file", "aliases", "roles_by_sheet"]:
      if k in st.session_state:
          st.session_state.pop(k, None)
    st.rerun()

def _login_view():
    render_logo_topright(LOGO_SIZE_LOGIN)
    st.markdown("## 🔐 Iniciar sesión")
    with st.form("login_form"):
        u = st.text_input("Usuario", value="")
        p = st.text_input("Contraseña", type="password", value="")
        ok = st.form_submit_button("Entrar")
        if ok:
            if (u or "") == str(USER) and (p or "") == str(PASSWORD):
                st.session_state.authenticated = True
                st.success("Acceso concedido. Cargando…")
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

# Botón de cerrar sesión solo si ya inició
if st.session_state.authenticated:
    st.sidebar.button("🚪 Cerrar sesión", on_click=_logout, use_container_width=True)

# Gate de acceso: si no está autenticado, muestra login y detiene la renderización
if not st.session_state.authenticated:
    _login_view()
    st.stop()

# Logo en la vista principal (arriba a la derecha)
render_logo_topright(LOGO_SIZE_APP)

# =========================
# 🔐 Fin Login + Logo
# =========================

# ======== Estilo ========
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

ss = st.session_state
ss.setdefault("historial", [])
ss.setdefault("data", None)
ss.setdefault("sheet_url", "")
ss.setdefault("max_cats_grafico", 18)
ss.setdefault("top_n_grafico", 12)
ss.setdefault("aliases", {})
ss.setdefault("menu_sel", "KPIs")
ss.setdefault("roles_forced", {})
ss.setdefault("_wkey", 0)

# ======== Carga de datos ========
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

# ======== OpenAI ========
def _get_openai_client():
    api_key = (st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key") or
               os.getenv("OPENAI_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not api_key: return None
    api_key = str(api_key).strip().strip('"').strip("'")
    org = st.secrets.get("OPENAI_ORG") or os.getenv("OPENAI_ORG")
    base_url = st.secrets.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    kw = {"api_key": api_key}
    if org: kw["organization"] = str(org).strip()
    if base_url: kw["base_url"] = str(base_url).strip()
    try: return OpenAI(**kw)
    except: return None

def ask_gpt(messages) -> str:
    client = _get_openai_client()
    if client is None:
        return "⚠️ No hay OPENAI_API_KEY configurada."
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.15
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Fallo OpenAI: {e}")
        return "⚠️ Error de IA."

def diagnosticar_openai():
    res = {"api_key_present": False, "organization_set": False, "base_url_set": False,
           "list_models_ok": False, "chat_ok": False, "quota_ok": None, "usage_tokens": None, "error": None}
    key = (st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY"))
    res["api_key_present"] = bool(key)
    res["organization_set"] = bool(st.secrets.get("OPENAI_ORG") or os.getenv("OPENAI_ORG"))
    res["base_url_set"] = bool(st.secrets.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    if not res["api_key_present"]:
        res["error"] = "Falta OPENAI_API_KEY."
        return res
    client = _get_openai_client()
    if client is None:
        res["error"] = "No pude inicializar el cliente."
        return res
    try:
        _ = client.models.list()
        res["list_models_ok"] = True
    except Exception as e:
        res["error"] = f"Error listando modelos: {e}"
        return res
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":"Ping diagnóstico. Responde OK."}],
            temperature=0, max_tokens=5
        )
        res["chat_ok"] = True
        if getattr(r, "usage", None) and getattr(r.usage, "total_tokens", None) is not None:
            res["usage_tokens"] = int(r.usage.total_tokens)
        if res["quota_ok"] is None: res["quota_ok"] = True
    except Exception as e:
        res["error"] = f"Error en chat: {e}"
        res["quota_ok"] = None
    return res

# ======== Limpieza / formato ========
ALL_SPACES_RE = re.compile(r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]')
INVISIBLES_RE = re.compile(r'[\u200B\u200C\u200D\uFEFF\u2060\u00AD]')
TITLE_LINE_RE = re.compile(r'^(#{1,6}\s+[^\n]+)$', re.M)
CURRENCY_COMMA_RE = re.compile(r'(?<![\w$])(\d{1,3}(?:,\d{3})+)(?![\w%])')
CURRENCY_DOT_RE   = re.compile(r'(?<![\w$])(\d{1,3}(?:\.\d{3})+)(?![\w%])')
MILLONES_RE = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*millone?s\b')
MILES_RE    = re.compile(r'(?i)\$?\s*(\d+(?:[.,]\d+)?)\s*mil\b')
ARTIFACT_DOLLAR_BETWEEN_GROUPS = re.compile(r'(?<=[\.,])\$(?=\d{3}\b)')

def fmt_money(x):
    try: return f"${int(round(float(x))):,}".replace(",", ".")
    except: return str(x)

def fix_peso_artifacts(t: str) -> str:
    if not t: return t
    t = ARTIFACT_DOLLAR_BETWEEN_GROUPS.sub('', t)
    t = re.sub(r'\$\s*\$+', '$', t)
    t = re.sub(r'\$\s+(?=\d)', '$', t)
    t = re.sub(r'\$(\d{1,3}(?:[.,]\d{3})+)[.,]\d{1,2}\b', r'$\1', t)
    return t

def _space_punct_outside_numbers(s: str) -> str:
    s = re.sub(r',(?=[^\d\s])', ', ', s)
    s = re.sub(r':(?=[^\s])', ': ', s)
    return s

def prettify_answer(text: str) -> str:
    if not text: return text
    t = unicodedata.normalize("NFKC", text)
    t = unescape(t)
    t = re.sub(r'<[^>]+>', '', t)
    t = INVISIBLES_RE.sub('', t)
    t = ALL_SPACES_RE.sub(' ', t)
    t = t.replace("•", "\n- ").replace("“","\"").replace("”","\"").replace("’","'")
    t = re.sub(r'([*_`~]{1,3})(?=\S)(.+?)(?<=\S)\1', r'\2', t)
    t = TITLE_LINE_RE.sub(r'\1\n', t)
    t = re.sub(r'^[\s]*[-•]\s*', '- ', t, flags=re.M)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n\s+', '\n', t)
    t = re.sub(r'(?i)\bmargende\b', 'margen de', t)
    t = re.sub(r'(?i)\bmientrasque\b', 'mientras que', t)
    t = re.sub(r'(?i)(?:US|CLP|COP|S)\s*\$', '$', t)
    def _mill(m):
        try: return fmt_money(int(round(float(m.group(1).replace(",","."))*1_000_000)))
        except: return m.group(0)
    def _mil(m):
        try: return fmt_money(int(round(float(m.group(1).replace(",","."))*1_000)))
        except: return m.group(0)
    t = MILLONES_RE.sub(_mill, t); t = MILES_RE.sub(_mil, t)
    t = CURRENCY_COMMA_RE.sub(lambda m: fmt_money(int(m.group(1).replace(',',''))), t)
    t = CURRENCY_DOT_RE.sub(  lambda m: fmt_money(int(m.group(1).replace('.',''))), t)
    t = fix_peso_artifacts(t)
    t = _space_punct_outside_numbers(t)
    return t.strip()

def sanitize_text_for_html(s: str) -> str:
    if not s: return ""
    t = unicodedata.normalize("NFKC", s)
    t = unescape(t)
    t = re.sub(r'[\u200B\u200C\u200D\uFEFF\u2060\u00AD]', '', t)
    t = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]', ' ', t)
    t = re.sub(r'\\\((.*?)\\\)', r'\1', t, flags=re.S)
    t = re.sub(r'\\\[(.*?)\\\]', r'\1', t, flags=re.S)
    t = re.sub(r'\$\$(.*?)\$\$', r'\1', t, flags=re.S)
    t = fix_peso_artifacts(t)
    t = t.replace("•", "\n- ")
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n\s+','\n', t)
    t = _space_punct_outside_numbers(t)
    return t.strip()

def md_to_safe_html(markdown_text: str) -> str:
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

# ======== Ejes / Números ========
def _fmt_pesos(x, pos=None):
    try: return f"${int(round(float(x))):,}".replace(",", ".")
    except: return ""
def _fmt_int_axis(x, pos=None):
    try: return f"{int(round(float(x))):,}".replace(",", ".")
    except: return ""
def _fmt_number_general(x):
    try: return f"{int(round(float(x))):,}".replace(",", ".")
    except: return str(x)

# ======== Helpers ========
def _unique_key(prefix: str) -> str:
    ss._wkey += 1
    return f"{prefix}_{ss._wkey}"

def _export_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0); return buf.read()

def _norm(s: str) -> str:
    s = str(s).replace("\u00A0"," ").strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+',' ', s)
    return s.lower()

ID_PAT   = re.compile(r'(?i)\b(id|folio|factura|boleta|ot|orden|nro|n°|correlativo|documento|doc|num|numero|oc|os)\b')
MONEY_PAT= re.compile(r'(?i)(monto|valor|ingreso|ingresos|costo|costos|neto|bruto|precio|tarifa|pago|total|subtotal|margen|venta|ventas|compras)')
PCT_PAT  = re.compile(r'(?i)(%|porcentaje|tasa|margen %|margen_pct|conversion)')
DATE_PAT = re.compile(r'(?i)(fecha|emision|f_emision|periodo|mes|año|anio|fecha_factura|fecha_ot|fecha_ingreso|fecha_salida)')
QTY_PAT  = re.compile(r'(?i)(cantidad|unidades|servicios|items|piezas|qty|cant)')
CAT_HINT = re.compile(r'(?i)(tipo|clase|categoria|estado|proceso|servicio|cliente|patente|sucursal|marca|modelo|vehiculo|vehículo)')

def _ratio_unique(s: pd.Series) -> float:
    n = len(s)
    if n==0: return 0.0
    return float(s.nunique(dropna=True))/float(n)

def _is_int_like(series: pd.Series) -> bool:
    try:
        s = pd.to_numeric(series, errors="coerce")
        return bool(np.nanmax(np.mod(s,1)) == 0)
    except Exception:
        return False

def detect_roles_for_sheet(df: pd.DataFrame, sheet_name: str) -> Dict[str,str]:
    roles = {}
    for c in df.columns:
        name = str(c); nname = _norm(name); s = df[c]
        if DATE_PAT.search(nname) or "datetime" in str(s.dtype) or "date" in str(s.dtype):
            roles[name] = "date"; continue
        if PCT_PAT.search(nname): roles[name] = "percent"; continue
        if MONEY_PAT.search(nname): roles[name] = "money"; continue
        if ID_PAT.search(nname): roles[name] = "id"; continue
        if pd.api.types.is_numeric_dtype(s):
            uniq = _ratio_unique(s)
            try:
                mx = float(pd.to_numeric(s, errors="coerce").max())
                mn = float(pd.to_numeric(s, errors="coerce").min())
            except Exception:
                mx, mn = np.nan, np.nan
            if _is_int_like(s) and uniq > 0.80 and (mx > 10000 or (mx-mn) > 10000):
                roles[name] = "id"
            elif mx and mx >= 1000:
                roles[name] = "money"
            elif uniq < 0.20:
                roles[name] = "category"
            else:
                roles[name] = "text"
        else:
            roles[name] = "category" if (_ratio_unique(s) < 0.20 or CAT_HINT.search(nname)) else "text"
    for (hoja, colnorm), rol in ss.roles_forced.items():
        if _norm(hoja) == _norm(sheet_name):
            for c in list(roles.keys()):
                if _norm(c) == colnorm:
                    roles[c] = rol
    return roles

def find_col(df: pd.DataFrame, name: str):
    if not name: return None
    alias = ss.aliases.get(_norm(name))
    if alias and alias in df.columns: return alias
    tgt = _norm(name)
    for c in df.columns:
        if _norm(c) == tgt: return c
    candidates = [c for c in df.columns if tgt in _norm(c) or _norm(c).startswith(tgt[:4])]
    return candidates[0] if candidates else None

# ======== Visualizaciones ========
def mostrar_tabla(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False).reset_index())
    resumen.columns = [str(col_categoria).title(), str(col_valor).title()]
    col_val = resumen.columns[1]
    resumen[col_val] = resumen[col_val].apply(fmt_money)
    total_val = vals.sum(skipna=True)
    resumen.loc[len(resumen)] = ["TOTAL", fmt_money(total_val)]
    st.markdown(f"### 📊 {titulo if titulo else f'{col_val} por {col_categoria}'}")
    st.dataframe(resumen, use_container_width=True)
    st.download_button("⬇️ Descargar tabla (CSV)",
                       resumen.to_csv(index=False).encode("utf-8"),
                       "tabla.csv", "text/csv",
                       key=_unique_key("csv"))

def mostrar_tabla_count(g: pd.DataFrame, cat_col: str, count_col: str, titulo: str):
    df = g[[cat_col, count_col]].copy()
    df = df.sort_values(count_col, ascending=False)
    df.columns = ["Categoría", "OCs/OTs"]
    df["OCs/OTs"] = df["OCs/OTs"].fillna(0).astype(int)
    total = int(df["OCs/OTs"].sum())
    df_total = pd.DataFrame([{"Categoría":"TOTAL","OCs/OTs": total}])
    tabla = pd.concat([df, df_total], ignore_index=True)
    st.markdown(f"### 📋 {titulo}")
    st.dataframe(tabla, use_container_width=True)
    st.download_button("⬇️ Descargar tabla (CSV)",
                       tabla.to_csv(index=False).encode("utf-8"),
                       "tabla.csv", "text/csv",
                       key=_unique_key("csv"))

def _barras_vertical_generic(resumen, titulo, as_money=True):
    fig, ax = plt.subplots()
    bars = ax.bar(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo)
    ax.set_ylabel("Valor")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos if as_money else _fmt_int_axis))
    ax.tick_params(axis='x', rotation=45)
    for lbl in ax.get_xticklabels(): lbl.set_ha('right')
    for b in bars:
        y = b.get_height()
        if np.isfinite(y):
            label = (fmt_money(y) if as_money else _fmt_number_general(y))
            ax.annotate(label, (b.get_x()+b.get_width()/2, y), textcoords="offset points",
                        xytext=(0,3), ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    try:
        st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png",
                           key=_unique_key("png"))
    except Exception as e:
        st.caption(f"Descarga no disponible: {e}")

def _barras_horizontal_generic(resumen, titulo, as_money=True):
    fig, ax = plt.subplots()
    bars = ax.barh(resumen.index.astype(str), resumen.values)
    ax.set_title(titulo)
    ax.set_xlabel("Valor")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos if as_money else _fmt_int_axis))
    for b in bars:
        x = b.get_width()
        if np.isfinite(x):
            label = (fmt_money(x) if as_money else _fmt_number_general(x))
            ax.annotate(label, (x, b.get_y()+b.get_height()/2), textcoords="offset points",
                        xytext=(5,0), ha='left', va='center', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    try:
        st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png",
                           key=_unique_key("png"))
    except Exception as e:
        st.caption(f"Descarga no disponible: {e}")

def mostrar_grafico_barras_v3(df, col_categoria, col_valor, titulo=None):
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
    labels = [str(x) for x in resumen.index.tolist()]
    avg_len = float(np.mean([len(s) for s in labels])) if labels else 0.0
    if avg_len > 10:
        _barras_horizontal_generic(resumen, titulo or f"{col_valor} por {col_categoria}", True)
    else:
        _barras_vertical_generic(resumen, titulo or f"{col_valor} por {col_categoria}", True)

def mostrar_grafico_barras_count(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals)
                 .groupby(col_categoria, dropna=False)["__v"]
                 .sum()
                 .sort_values(ascending=False))
    labels = [str(x) for x in resumen.index.tolist()]
    avg_len = float(np.mean([len(s) for s in labels])) if labels else 0.0
    if avg_len > 10:
        _barras_horizontal_generic(resumen, titulo or f"Conteo por {col_categoria}", False)
    else:
        _barras_vertical_generic(resumen, titulo or f"Conteo por {col_categoria}", False)

def mostrar_grafico_torta(df, col_categoria, col_valor, titulo=None):
    vals = pd.to_numeric(df[col_valor], errors="coerce")
    resumen = (df.assign(__v=vals).groupby(col_categoria, dropna=False)["__v"]
                 .sum().sort_values(ascending=False))
    fig, ax = plt.subplots()
    ax.pie(resumen.values, labels=[str(x) for x in resumen.index], autopct='%1.1f%%', startangle=90)
    ax.axis('equal'); ax.set_title(titulo or f"{col_valor} por {col_categoria}")
    st.pyplot(fig)
    try:
        st.download_button("⬇️ PNG", _export_fig(fig), "grafico.png", "image/png",
                           key=_unique_key("png"))
    except Exception as e:
        st.caption(f"Descarga no disponible: {e}")

def mostrar_grafico_linea(df, x_col, y_col, titulo=None):
    vals = pd.to_numeric(df[y_col], errors="coerce")
    g = (df.assign(__v=vals).groupby(x_col, dropna=False)["__v"].sum().sort_index())
    fig, ax = plt.subplots()
    ax.plot(g.index.astype(str), g.values, marker="o")
    ax.set_title(titulo or f"{y_col} por {x_col}")
    ax.set_ylabel(y_col)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pesos))
    ax.tick_params(axis='x', rotation=45)
    for lbl in ax.get_xticklabels(): lbl.set_ha('right')
    fig.tight_layout()
    st.pyplot(fig)
    try:
        st.download_button("⬇️ PNG", _export_fig(fig), "serie.png", "image/png",
                           key=_unique_key("png"))
    except Exception as e:
        st.caption(f"Descarga no disponible: {e}")

# ======== Finanzas resumen ========
def _sheet_name_matches_finanzas(name: str) -> bool:
    n = _norm(name)
    return any(k in n for k in ["finanz", "finance"])

def _values_contain_keywords(series: pd.Series, keywords: List[str]) -> bool:
    try:
        vals = " ".join([str(x) for x in series.dropna().astype(str).unique()[:200]]).lower()
        return any(k in vals for k in keywords)
    except Exception:
        return False

def render_finance_table(data: Dict[str, pd.DataFrame]) -> None:
    target = None
    for h, df in data.items():
        if df is None or df.empty: continue
        if _sheet_name_matches_finanzas(h):
            target = (h, df); break
    if target:
        h, df = target
        roles = detect_roles_for_sheet(df, h)
        money_cols = [c for c,r in roles.items() if r=="money"]
        cat_cols   = [c for c,r in roles.items() if r=="category"]
        best = None
        for cat in cat_cols:
            if _values_contain_keywords(df[cat], ["ingreso","egreso","costo","costos"]):
                pick = None; pick_sum = -1
                for mcol in money_cols:
                    s = pd.to_numeric(df[mcol], errors="coerce").sum()
                    if s > pick_sum:
                        pick_sum = s; pick = mcol
                if pick:
                    best = (cat, pick); break
        if best:
            cat, val = best
            mostrar_tabla(df, cat, val, "Ingresos y Egresos (FINANZAS)")
            return
    st.info("No encontré una hoja FINANZAS estructurada; muestro KPIs.")
    try:
        k = analizar_datos_taller(data, "") or {}
    except Exception:
        k = {}
    ingresos = float(k.get("ingresos") or 0); costos = float(k.get("costos") or 0)
    margen = ingresos - costos
    df_tab = pd.DataFrame({"Concepto":["Ingresos","Costos","Margen"], "Monto":[ingresos,costos,margen]})
    df_tab["Monto"] = df_tab["Monto"].apply(fmt_money)
    st.dataframe(df_tab, use_container_width=True)

# ======== Pair finders ========
def find_best_pair_money(data: Dict[str, pd.DataFrame], category_match: callable):
    best = None
    for h, df in data.items():
        if df is None or df.empty: continue
        roles = detect_roles_for_sheet(df, h)
        money_cols = [c for c,r in roles.items() if r=="money"]
        if not money_cols: continue
        cat_cols = [c for c,r in roles.items() if r in ("category","text")]
        matches = [c for c in cat_cols if category_match(_norm(c))]
        for cat in matches:
            pick = None; pick_sum = -1
            for mcol in money_cols:
                s = pd.to_numeric(df[mcol], errors="coerce").sum()
                if s > pick_sum:
                    pick_sum = s; pick = mcol
            if pick is None: continue
            if best is None or pick_sum > best[-1]:
                best = (h, df, cat, pick, pick_sum)
    if best:
        h, df, cat, val, _ = best
        return h, df, cat, val
    return None, None, None, None

def find_best_pair_generic(data: Dict[str, pd.DataFrame], category_match: callable):
    h, df, cat, val = find_best_pair_money(data, category_match)
    if h: return h, df, cat, val, "sum"
    best = None
    for hh, ddf in data.items():
        if ddf is None or ddf.empty: continue
        roles = detect_roles_for_sheet(ddf, hh)
        cat_cols = [c for c,r in roles.items() if r in ("category","text")]
        matches = [c for c in cat_cols if category_match(_norm(c))]
        if not matches: continue
        num_cols = [c for c in ddf.select_dtypes(include=[np.number]).columns
                    if roles.get(c) not in ("percent","id")]
        for catc in matches:
            pick = None; pick_sum = -1
            for mcol in num_cols:
                s = pd.to_numeric(ddf[mcol], errors="coerce").sum()
                if s > pick_sum: pick_sum = s; pick = mcol
            if pick is None: continue
            if best is None or pick_sum > best[-1]:
                best = (hh, ddf, catc, pick, pick_sum)
    if best:
        hh, ddf, catc, pick, _ = best
        return hh, ddf, catc, pick, "sum"
    for hh, ddf in data.items():
        if ddf is None or ddf.empty: continue
        roles = detect_roles_for_sheet(ddf, hh)
        cat_cols = [c for c,r in roles.items() if r in ("category","text")]
        matches = [c for c in cat_cols if category_match(_norm(c))]
        if matches:
            return hh, ddf, matches[0], "__count__", "count"
    return None, None, None, None, None

def make_count_df_by_id(df: pd.DataFrame, cat_col: str, roles: Dict[str,str]) -> Optional[pd.DataFrame]:
    id_cols = [c for c,r in roles.items() if r=="id"]
    if not id_cols:
        g = df.groupby(cat_col, dropna=False).size().reset_index(name="__count__")
        return g
    idc = id_cols[0]
    d = df[[cat_col, idc]].dropna(subset=[idc])
    d = d.drop_duplicates([cat_col, idc])
    g = d.groupby(cat_col, dropna=False)[idc].nunique().reset_index(name="__count__")
    return g

def best_count_by_category(data: Dict[str,pd.DataFrame], category_match: callable):
    best = None
    for h, df in data.items():
        if df is None or df.empty: continue
        roles = detect_roles_for_sheet(df, h)
        cats = [c for c,r in roles.items() if r in ("category","text")]
        matches = [c for c in cats if category_match(_norm(c))]
        if not matches: continue
        cat = matches[0]
        g = make_count_df_by_id(df, cat, roles)
        if g is None or g.empty: continue
        total = int(g["__count__"].sum())
        if best is None or total > best[-1]:
            best = (h, df, cat, g, total)
    if best:
        h, df, cat, g, _ = best
        return h, df, cat, g
    return None, None, None, None

# ======== Insights & textos ========
def derive_global_insights(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        kpis = analizar_datos_taller(data, "") or {}
    except Exception:
        kpis = {}
    best = None
    for h, df in data.items():
        if df is None or df.empty: continue
        roles = detect_roles_for_sheet(df, h)
        pcol = None
        for name in ["proceso","tipo de proceso","servicio","servicios","estado"]:
            if find_col(df, name): pcol = find_col(df, name); break
        vcol = None
        for c,r in roles.items(): 
            if r=="money": vcol = c; break
        if not pcol or not vcol: continue
        total = float(pd.to_numeric(df[vcol], errors="coerce").sum())
        if best is None or total > best["total"]:
            best = {"sheet": h, "pcol": pcol, "vcol": vcol, "total": total}
    by_process, conc_share = [], None
    if best:
        h, pcol, vcol = best["sheet"], best["pcol"], best["vcol"]
        d = data[h]
        vals = pd.to_numeric(d[vcol], errors="coerce")
        s = (d.assign(__v=vals).groupby(pcol, dropna=False)["__v"].sum().sort_values(ascending=False))
        total = float(s.sum()) if len(s) else 0.0
        by_process = [{"proceso": str(k), "monto": float(v) if pd.notna(v) else 0.0} for k, v in s.items()]
        if total > 0 and len(by_process) >= 1:
            conc_share = by_process[0]["monto"]/total
        sheet_for_process = h
    else:
        sheet_for_process = None
    ingresos = float(kpis.get("ingresos") or 0); costos = float(kpis.get("costos") or 0)
    margen = ingresos - costos; margen_pct = (margen/ingresos) if ingresos>0 else None
    lt = kpis.get("lead_time_mediano_dias"); conv = kpis.get("conversion_pct")
    alerts, opps = [], []
    if conc_share is not None and conc_share >= 0.60:
        alerts.append(f"Alta concentración: el proceso líder representa {conc_share*100:.1f}% del total.")
        if len(by_process) >= 2: opps.append(f"Diversificar hacia '{by_process[1]['proceso']}'.")
    margen_target = 0.10
    if margen_pct is not None and margen_pct < margen_target and ingresos>0:
        ventas_nec = costos/(1-margen_target); uplift = max(0.0, ventas_nec/ingresos-1.0)
        opps.append(f"Ajustes de precios: subir {uplift*100:.1f}% para margen objetivo {margen_target*100:.1f}%.")
        alerts.append(f"Margen bajo: actual {margen_pct*100:.1f}% sobre {fmt_money(ingresos)}.")
    if lt and lt>4.5: opps.append("Reducir lead time 1 día eleva capacidad ~+33%.")
    if conv and conv<75: opps.append("Subir conversión 5–10 pp con playbook de cierre.")
    proj = {"base": ingresos, "optim": ingresos*(1+0.05), "cons": ingresos*0.95}
    return {"kpis":kpis,"by_process":by_process,"process_concentration":conc_share,
            "alerts":alerts,"opportunities":opps,"targets":{"margin_pct":margen_target},
            "projection_6m":proj,"sheet_for_process":sheet_for_process,"by_process_role":"money"}

def compose_actionable_text(ins: Dict[str,Any]) -> str:
    k = ins.get("kpis", {})
    ingresos = float(k.get("ingresos") or 0); costos = float(k.get("costos") or 0)
    margen   = ingresos - costos
    margen_pct = (margen/ingresos) if ingresos>0 else None
    bp = ins.get("by_process", [])
    proj = ins.get("projection_6m", {})
    tgt = ins.get("targets", {}).get("margin_pct", 0.10)

    lines = ["### Diagnóstico basado en datos"]
    if ingresos:
        lines.append(f"- Ingresos: {fmt_money(ingresos)}; costos: {fmt_money(costos)}; margen: {fmt_money(margen)} ({margen_pct*100:.1f}%).")
    if bp:
        top = bp[0]; share = ins.get("process_concentration")
        if share is not None:
            lines.append(f"- Proceso líder: **{top['proceso']}** con {fmt_money(top['monto'])} ({share*100:.1f}%).")
        else:
            lines.append(f"- Proceso líder: **{top['proceso']}** con {fmt_money(top['monto'])}.")
    for a in ins.get("alerts", []): lines.append(f"- ⚠️ {a}")

    recs = []
    if margen_pct is not None and margen_pct < tgt and ingresos>0:
        ventas_nec = costos/(1-tgt); uplift = max(0.0, ventas_nec/ingresos - 1.0)
        recs.append(f"**Ajuste de precios**: subir **{uplift*100:.1f}%** para alcanzar margen de **{tgt*100:.1f}%**.")
    if ins.get("process_concentration") and ins["process_concentration"]>=0.60 and len(bp)>=2:
        recs.append(f"**Diversificar mix**: empujar desde **{bp[0]['proceso']}** a **{bp[1]['proceso']}**.")
    lt = k.get("lead_time_mediano_dias")
    if lt and lt>4.5: recs.append("**Operaciones**: bajar 1 día el lead time (capacidad ~+33%).")
    conv = k.get("conversion_pct")
    if conv and conv<75: recs.append("**Comercial**: subir 5–10 pp la conversión (playbook de cierre).")
    if costos>0: recs.append("**Compras**: renegociar insumos clave (meta -3% costo promedio).")

    secciones = []
    secciones.append("\n".join(lines))
    secciones.append("### Recomendaciones de gestión (priorizadas)\n" + "\n".join([f"{i+1}. {r}" for i,r in enumerate(recs)]))

    if proj:
        base = proj.get("base", ingresos); optim = proj.get("optim", ingresos); cons = proj.get("cons", ingresos)
        secciones.append("### Estimaciones y proyecciones (6 meses)\n"
                         f"- **Base**: {fmt_money(base)} / mes.\n"
                         f"- **Optimista**: {fmt_money(optim)} / mes.\n"
                         f"- **Conservador**: {fmt_money(cons)} / mes.")
    secciones.append("### Próximos pasos\n- Finanzas: simulación de margen y precios.\n- Operaciones: plan de reducción de lead time.\n- Comercial: campaña de mix.")
    return "\n\n".join(secciones)

def compose_operational_text(data: Dict[str, Any]) -> str:
    estados_hints = ["estado","etapa","workflow","fase","situacion","situación"]
    fecha_hints = ["fecha","emision","emisión","fecha_ot","fecha_ingreso","fecha_entrega"]
    total_reg = 0
    bloques = ["### Diagnóstico operacional"]
    for h, df in data.items():
        if df is None or df.empty: continue
        roles = detect_roles_for_sheet(df, h)
        estado_col = None
        for c in df.columns:
            if roles.get(c) in ("category","text"):
                if any(k in _norm(c) for k in estados_hints):
                    estado_col = c; break
        if not estado_col: continue
        total_reg += len(df)
        cnt = df[estado_col].astype(str).str.upper().value_counts(dropna=False).head(8)
        txt = "; ".join([f"{k}: {v}" for k,v in cnt.items()])
        bloques.append(f"- **{h}** → WIP por estado: {txt}")
        fechas = [c for c,r in roles.items() if r=="date" and any(k in _norm(c) for k in fecha_hints)]
        if len(fechas)>=2:
            d1 = pd.to_datetime(df[fechas[0]], errors="coerce")
            d2 = pd.to_datetime(df[fechas[1]], errors="coerce")
            lt = (d2 - d1).dt.days
            if lt.notna().any():
                bloques.append(f"  · Lead time medio: {lt.mean():.1f} días (p50={lt.median():.1f}).")
    if total_reg==0:
        bloques.append("- No encontré columnas de **estado** para operaciones.")
    bloques.append("\n### Recomendaciones operativas")
    bloques += [
        "1. Balancear carga entre etapas críticas (donde haya más WIP).",
        "2. Daily management: carril rápido para OT simples (reduce lead time cola).",
        "3. Control visual de estado (tablero) y SLA por etapa.",
        "4. Causa-raíz de re-trabajos y tiempos muertos (quick wins en 2 semanas)."
    ]
    return "\n".join(bloques)

def compose_market_text(data: Dict[str, Any]) -> str:
    def is_marca(n):  return any(k in n for k in ("marca","brand","make","fabricante"))
    def is_modelo(n): return any(k in n for k in ("modelo","model","version","versión","trim"))

    def top_from_count(match_fn):
        h, df, cat, g = best_count_by_category(data, match_fn)
        if not h: return []
        g = g.sort_values("__count__", ascending=False).head(5)
        return [(str(r[cat]), int(r["__count__"])) for _, r in g.iterrows()]

    top_marca  = top_from_count(is_marca)
    top_modelo = top_from_count(is_modelo)

    estac_txt = "—"
    for h, df in data.items():
        if df is None or df.empty: continue
        roles = detect_roles_for_sheet(df, h)
        date_cols = [c for c,r in roles.items() if r=="date"]
        if not date_cols: continue
        dt = pd.to_datetime(df[date_cols[0]], errors="coerce")
        d2 = df.assign(__mm = dt.dt.to_period("M").dt.to_timestamp(),
                       MES = lambda d: d["__mm"].dt.strftime("%Y-%m"))
        money_cols= [c for c,r in roles.items() if r=="money"]
        if money_cols:
            s = (d2.assign(__v=pd.to_numeric(d2[money_cols[0]], errors="coerce"))
                    .groupby("MES")["__v"].sum().sort_values(ascending=False))
            if len(s)>0:
                estac_txt = f"mes pico: {s.index[0]} ({fmt_money(float(s.iloc[0]))})"
                break
        else:
            s = d2.groupby("MES").size().sort_values(ascending=False)
            if len(s)>0:
                estac_txt = f"mes pico: {s.index[0]} ({_fmt_number_general(int(s.iloc[0]))} OCs)"
                break

    lines = ["### Insights de mercado",
             "- Mix de clientes y productos, con focos comerciales accionables.",
             "\n**Top marcas (por OCs/OTs)**:"]
    if top_marca:
        lines += [f"- {i+1}. {n}: {_fmt_number_general(v)}" for i,(n,v) in enumerate(top_marca)]
    else:
        lines.append("- No se detectaron marcas.")

    lines.append("\n**Top modelos (por OCs/OTs)**:")
    if top_modelo:
        lines += [f"- {i+1}. {n}: {_fmt_number_general(v)}" for i,(n,v) in enumerate(top_modelo)]
    else:
        lines.append("- No se detectaron modelos.")

    lines.append("\n**Estacionalidad:** " + estac_txt)

    lines.append("\n### Recomendaciones de marketing")
    lines += [
        "1. Campaña lookalike sobre clientes del segmento líder (email + WhatsApp).",
        "2. Paquetes por modelo top (precio ancla + upsell de detailing).",
        "3. Promos anticíclicas 2 semanas antes del mes con menor demanda.",
        "4. Cross-sell a clientes de 'desarme' hacia 'pintura' con voucher del 5%.",
        "5. Landing con testimonios para marcas top (SEO local por marca/modelo)."
    ]
    return "\n".join(lines)

# ======== Prompts LLM ========
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
Con base en los KPIs reales, realiza un ANÁLISIS PROFESIONAL con el formato obligatorio.
No inventes datos fuera de lo entregado; si necesitas supuestos, decláralos.

KPIs:
{json.dumps(analisis_dict, ensure_ascii=False, indent=2)}

{ANALYSIS_FORMAT}
"""}
    ]

def _build_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    schema = {}
    for hoja, df in data.items():
        if df is None or df.empty: continue
        roles = detect_roles_for_sheet(df, hoja)
        cols = []; samples = {}
        for c in df.columns:
            cols.append(str(c))
            vals = df[c].dropna().astype(str).head(3).tolist()
            if vals: samples[str(c)] = vals
        schema[hoja] = {"columns": cols, "samples": samples, "roles": roles}
    return schema

def prompt_consulta_libre(pregunta: str, schema: dict) -> list:
    focus = detect_focus_from_question(pregunta)["focus"]
    historial_msgs = []
    for h in ss.historial[-6:]:
        historial_msgs += [{"role":"user","content":h["pregunta"]},
                           {"role":"assistant","content":h["respuesta"]}]
    system = make_system_prompt()
    user = f"""
Responde SOLO sobre el tema **{focus}** detectado en la pregunta.
- NO repitas tablas ni KPIs tal como aparecen; **complementa** con insights de ese tema.
- Si incluyes `viz:` debe ser SOLO UNA y coherente con **{focus}**.
- Si el usuario pide por **mes/año**, usa una columna con rol 'date' y agrupa por **month** o **year** (lo que mejor corresponda).

Esquema (con roles):
{json.dumps(schema, ensure_ascii=False, indent=2)}

Pregunta:
{pregunta}

{ANALYSIS_FORMAT}
"""
    return [{"role":"system","content":system}, *historial_msgs, {"role":"user","content":user}]

def detect_focus_from_question(q: str) -> dict:
    qn = _norm(q)
    if any(k in qn for k in ["patente", "patentes", "dominio"]):
        return {"focus": "patentes", "cat_hints": ["patente", "patentes", "dominio"]}
    if "cliente" in qn:
        return {"focus": "clientes", "cat_hints": ["tipo de cliente", "tipo_cliente", "cliente", "segmento"]}
    if any(k in qn for k in ["proceso", "servicio", "operacion", "operación", "recepcion", "reparacion", "reparación"]):
        return {"focus": "procesos", "cat_hints": ["proceso", "tipo de proceso", "servicio", "servicios", "estado"]}
    if any(k in qn for k in ["mes", "enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]):
        return {"focus": "tiempo", "time": "month"}
    if "año" in qn or "anio" in qn or "2024" in qn or "2025" in qn:
        return {"focus": "tiempo", "time": "year"}
    if "finanz" in qn or "factur" in qn or "ingreso" in qn or "egreso" in qn or "costo" in qn:
        return {"focus": "finanzas", "cat_hints": ["categoria", "tipo", "glosa"]}
    return {"focus": "general", "cat_hints": []}

def plan_from_llm(pregunta: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    system = "Eres un controller financiero experto. Devuelve SOLO JSON."
    prompt = f"""
Devuelve SOLO un JSON para visualización:
{{ "action":"chart|table|text", "sheet":"", "category_col":"", "value_col":"", "chart":"barras|torta|auto", "title":"" }}

Reglas:
- Usa nombres EXACTOS del esquema (insensible a mayúsculas).
- Evita columnas rol 'id' como value_col (usa conteo si no hay otra).
- Si value_col es 'percent', considera mostrar promedio.
- Si la pregunta pide meses/años, probablemente convenga usar un eje temporal.

ESQUEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

PREGUNTA:
{pregunta}
"""
    raw = ask_gpt([{"role":"system","content":system},{"role":"user","content":prompt}]).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except Exception: return {}

def plan_compute_from_llm(pregunta: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    system = ("Eres un planificador de CÓMPUTO. Devuelve SOLO JSON. "
              "Evita 'id' como value_col; si sólo hay 'id' usa op='count'. "
              "Si value_col es 'percent', usa op='avg'. Si la pregunta menciona meses o años, "
              "elige una columna 'date' y define group_by como 'month' o 'year'.")
    prompt = f"""
Formato JSON ÚNICO:
{{
  "sheet": "<hoja|vacío>",
  "value_col": "<columna_de_valor>",
  "category_col": "<columna_categoria_o_fecha_o_vacío>",
  "op": "sum|avg|count|max|min",
  "filters": [{{"col":"<col>","op":"eq|contains|gte|lte","val":"<valor>"}}],
  "group_by": "none|month|year"
}}

ESQUEMA (con roles):
{json.dumps(schema, ensure_ascii=False, indent=2)}

PREGUNTA:
{pregunta}
"""
    raw = ask_gpt([{"role":"system","content":system},{"role":"user","content":prompt}]).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except Exception: return {}

# ======== Compute & Execute ========
def _apply_filters(df: pd.DataFrame, filters: List[Dict[str,str]]) -> pd.DataFrame:
    if not filters: return df
    out = df.copy()
    for f in filters:
        col = find_col(out, f.get("col",""))
        if not col: continue
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

def execute_plan(plan: Dict[str, Any], data: Dict[str, Any]) -> bool:
    action = (plan or {}).get("action")
    if action not in ("table", "chart"): return False
    sheet = plan.get("sheet") or ""
    cat = plan.get("category_col") or ""
    val = plan.get("value_col") or ""
    chart = (plan.get("chart") or "auto").lower()
    title = plan.get("title") or None
    hojas = [sheet] if sheet in data else list(data.keys())
    for h in hojas:
        df = data[h]
        if df is None or df.empty: continue
        roles = detect_roles_for_sheet(df, h)
        cat_real = find_col(df, cat) if cat else None
        val_real = find_col(df, val) if val else None
        if not (cat_real and val_real): continue
        if roles.get(val_real) == "id":
            st.info("La columna de valor es 'id'; muestro recuento por categoría.")
            mostrar_grafico_barras_count(df.assign(__count__=1), cat_real, "__count__", title or "Recuento")
            return True
        try:
            if action == "table":
                mostrar_tabla(df, cat_real, val_real, title); return True
            if chart in ("auto","barras"):
                mostrar_grafico_barras_v3(df, cat_real, val_real, title); return True
            if chart == "torta":
                mostrar_grafico_torta(df, cat_real, val_real, title); return True
        except Exception as e:
            st.error(f"Error visualizando en '{h}': {e}"); return False
    return False

def execute_compute(plan_c: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    if not plan_c: return {"ok": False, "msg": "No hubo plan de cómputo."}
    sheet = plan_c.get("sheet") or ""
    hojas = [sheet] if (sheet and sheet in data) else list(data.keys())
    group_by = (plan_c.get("group_by") or "none").lower()
    last_err = None
    for h in hojas:
        df = data[h]
        if df is None or df.empty: continue
        roles = detect_roles_for_sheet(df, h)

        vraw = plan_c.get("value_col")
        if not vraw:
            money = [c for c,r in roles.items() if r=="money"]
            vraw = money[0] if money else None
        if not vraw:
            last_err = "No se encontró columna de valor adecuada."
            continue
        vcol = find_col(df, vraw) or vraw
        if vcol not in df.columns:
            last_err = f"Columna de valor no existe en '{h}'."
            continue

        crow = plan_c.get("category_col") or ""
        ccol = find_col(df, crow) if crow else None
        filters = plan_c.get("filters") or []
        op = (plan_c.get("op") or "sum").lower()

        vrole = roles.get(vcol, "unknown")
        if vrole == "id" and op != "count": op = "count"

        dff = _apply_filters(df, filters) if filters else df
        try:
            if ccol and roles.get(ccol) == "date" and group_by in ("month","year"):
                _dt = pd.to_datetime(dff[ccol], errors="coerce")
                if group_by == "month":
                    dff = dff.assign(__grp=_dt.dt.to_period("M").dt.to_timestamp())
                    grp_label = dff["__grp"].dt.strftime("%Y-%m")
                else:
                    dff = dff.assign(__grp=_dt.dt.to_period("Y").dt.to_timestamp())
                    grp_label = dff["__grp"].dt.strftime("%Y")
                ccol = "__GRUPO_TIEMPO__"
                dff = dff.assign(**{ccol: grp_label})

            if ccol:
                if op == "count":
                    if vrole == "id":
                        srs = dff.groupby(ccol, dropna=False)[vcol].nunique()
                    else:
                        srs = dff.groupby(ccol, dropna=False)[vcol].count()
                elif op == "avg" or vrole == "percent":
                    srs = pd.to_numeric(dff[vcol], errors="coerce")
                    srs = dff.assign(__v=srs).groupby(ccol, dropna=False)["__v"].mean()
                elif op == "max":
                    srs = pd.to_numeric(dff[vcol], errors="coerce")
                    srs = dff.assign(__v=srs).groupby(ccol, dropna=False)["__v"].max()
                elif op == "min":
                    srs = pd.to_numeric(dff[vcol], errors="coerce")
                    srs = dff.assign(__v=srs).groupby(ccol, dropna=False)["__v"].min()
                else:
                    srs = pd.to_numeric(dff[vcol], errors="coerce")
                    srs = dff.assign(__v=srs).groupby(ccol, dropna=False)["__v"].sum()
                srs = srs.sort_values(ascending=False)
                total = float(srs.sum()) if op != "count" else int(srs.sum())
                df_res = srs.reset_index().rename(columns={ccol:"CATEGORIA", 0:"VALOR", "__v":"VALOR"})
                by_cat = [{"categoria": str(k), "valor": (float(v) if op!="count" else int(v))} for k,v in srs.items()]
                vrole_out = ("quantity" if op=="count" or vrole in ("id","quantity") else vrole)
                return {"ok": True, "sheet": h, "value_col": vcol, "category_col": ccol, "op": op,
                        "rows": int(len(dff)), "total": total, "by_category": by_cat,
                        "value_role": vrole_out, "df_result": df_res}
            else:
                if op == "count":
                    total = int(dff[vcol].nunique()) if vrole=="id" else int(dff[vcol].count())
                elif op == "avg" or vrole == "percent":
                    total = float(pd.to_numeric(dff[vcol], errors="coerce").mean())
                elif op == "max":
                    total = float(pd.to_numeric(dff[vcol], errors="coerce").max())
                elif op == "min":
                    total = float(pd.to_numeric(dff[vcol], errors="coerce").min())
                else:
                    total = float(pd.to_numeric(dff[vcol], errors="coerce").sum())
                vrole_out = ("quantity" if op=="count" or vrole in ("id","quantity") else vrole)
                df_res = pd.DataFrame({"CATEGORIA": ["TOTAL"], "VALOR": [total]})
                return {"ok": True, "sheet": h, "value_col": vcol, "category_col": None, "op": op,
                        "rows": int(len(dff)), "total": total, "by_category": [],
                        "value_role": vrole_out, "df_result": df_res}
        except Exception as e:
            last_err = f"Cálculo falló en '{h}': {e}"
            continue
    return {"ok": False, "msg": last_err or "No se pudo calcular."}

def build_verified_summary(facts: dict) -> str:
    val = facts.get("value_col","valor")
    cat = facts.get("category_col","")
    op   = facts.get("op","sum")
    total = facts.get("total",0); rows = facts.get("rows",0)
    encabezado = "### Resumen ejecutivo\n"
    def _fmt(v): 
        if facts.get("value_role")=="money" and op!="count": return fmt_money(v)
        return _fmt_number_general(v)
    if cat and facts.get("by_category"):
        lines = [f"- {val.title()} ({op.upper()}): total {_fmt(total)} por **{cat}** (sobre {rows} filas)."]
    else:
        lines = [f"- {val.title()} ({op.upper()}): {_fmt(total)} (sobre {rows} filas)."]
    return encabezado + "\n".join(lines)

def compose_focus_text(facts: dict, pregunta: str) -> str:
    op     = (facts.get("op") or "sum").upper()
    vrole  = facts.get("value_role","unknown")
    vcol   = facts.get("value_col","valor")
    cat    = facts.get("category_col")
    rows   = facts.get("rows",0)
    total  = facts.get("total",0)
    by_cat = facts.get("by_category", [])
    def _fmt(v):
        if vrole=="money" and op!="COUNT": return fmt_money(v)
        return _fmt_number_general(v)
    lines = [build_verified_summary(facts)]
    lines.append("### Diagnóstico")
    if cat and by_cat:
        top3 = by_cat[:3]
        det = "; ".join([f"{c['categoria']}: {_fmt(c['valor'])}" for c in top3])
        lines.append(f"- Top 3 por **{cat}** → {det}.")
        if len(by_cat) >= 5:
            tail = by_cat[3:5]
            det2 = "; ".join([f"{c['categoria']}: {_fmt(c['valor'])}" for c in tail])
            lines.append(f"- Siguientes → {det2}.")
    else:
        lines.append(f"- Resultado global: {_fmt(total)}.")
    recs = []
    if vrole=="money" and op in ("SUM","MAX"):
        recs.append("Ajustar mix hacia categorías con mejor margen.")
        recs.append("Revisar pricing en categorías con menor aporte.")
    elif op=="COUNT":
        recs.append("Diseñar campañas sobre las 3 categorías más frecuentes.")
        recs.append("Apalancar referidos en categorías con mayor recurrencia.")
    else:
        recs.append("Monitorizar tendencia en el tiempo para validar estabilidad.")
    lines.append("### Recomendaciones\n" + "\n".join([f"- {r}" for r in recs]))
    return "\n\n".join(lines)

# ======== Interfaz ========
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

    with st.expander("🔧 Diagnóstico del código"):
        st.caption(f"Build: **{APP_BUILD}**")
        try:
            h = hashlib.sha256(render_finance_table.__code__.co_code).hexdigest()[:16]
            st.caption(f"Hash finanzas: `{h}`")
        except Exception as e:
            st.caption(f"No pude inspeccionar funciones: {e}")

# ======== Vistas ========
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

elif ss.menu_sel == "Vista previa":
    data = ss.data
    if not data:
        st.info("Carga datos en **Datos**.")
    else:
        st.markdown("### 📄 Hojas (con roles detectados)")
        for name, df in data.items():
            st.markdown(f"#### 📘 {name} • filas: {len(df)}")
            roles = detect_roles_for_sheet(df, name)
            st.caption("Roles: " + ", ".join([f"`{c}`→{r}" for c,r in roles.items()]))
            st.dataframe(df.head(10), use_container_width=True)

elif ss.menu_sel == "KPIs":
    data = ss.data
    if not data:
        st.info("Carga datos en **Datos**.")
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
        c6.metric("Ticket promedio", f"{fmt_money(tp) if tp else '—'}")
        conv = kpis.get("conversion_pct")
        c7.metric("Conversión",   f"{conv:.1f}%" if conv is not None else "—")
        lt = kpis.get("lead_time_mediano_dias")
        if lt is not None: st.caption(f"⏱️ Lead time mediano: {lt:.1f} días")

elif ss.menu_sel == "Consulta IA":
    data = ss.data
    if not data:
        st.info("Carga datos en **Datos**.")
    else:
        st.markdown("### 🤖 Consulta")

        # Botones de análisis (arriba)
        cBtns = st.columns(4)
        left, right = st.columns([0.58, 0.42])

        # Línea de pregunta + responder (juntos)
        qcol, bcol = st.columns([0.82, 0.18])
        pregunta = qcol.text_input("Pregunta", label_visibility="collapsed",
                                   placeholder="Escribe tu pregunta...")
        responder_click = bcol.button("Responder")

        if cBtns[0].button("📊 Análisis General Automático"):
            analisis = analizar_datos_taller(data, "")
            ins = derive_global_insights(data)
            texto_extra = compose_actionable_text(ins)
            raw = ask_gpt(prompt_analisis_general(analisis))
            with left:
                render_ia_html_block(prettify_answer(raw) + "\n\n" + texto_extra, height=520)
            with right:
                render_finance_table(data)
                st.markdown("### Distribución por cliente y procesos")
                def is_tipo_cliente(n): return ("cliente" in n) and (("tipo" in n) or ("segment" in n))
                h1, df1, cat1, val1 = find_best_pair_money(data, is_tipo_cliente)
                def is_proceso(n): return (("proceso" in n) or ("servicio" in n))
                h2, df2, cat2, val2 = find_best_pair_money(data, is_proceso)
                cA, cB = st.columns(2)
                if h1 and df1 is not None:
                    with cA: mostrar_grafico_torta(df1, cat1, val1, "Distribución por Tipo de Cliente")
                else:
                    with cA: st.info("No encontré dinero + 'Tipo de cliente' para graficar.")
                if h2 and df2 is not None:
                    with cB: mostrar_grafico_barras_v3(df2, cat2, val2, "Monto por Proceso")
                else:
                    with cB: st.info("No encontré dinero + 'Proceso' para graficar.")
            ss.historial.append({"pregunta":"Análisis general","respuesta":raw})

        if cBtns[1].button("💵 Análisis Financiero"):
            ins = derive_global_insights(data)
            texto = "### Análisis financiero (foco)\n" + compose_actionable_text(ins)
            with left:  render_ia_html_block(texto, height=520)
            with right: render_finance_table(data)

        if cBtns[2].button("⚙️ Análisis Operacional"):
            texto = compose_operational_text(data)
            with left:  render_ia_html_block(texto, height=520)
            with right:
                def is_proceso(n): return ("proceso" in n) or ("servicio" in n)
                h, df, cat, val = find_best_pair_money(data, is_proceso)
                if h: mostrar_grafico_barras_v3(df, cat, val, "Monto por Proceso")
                else: st.info("No encontré un par (proceso, monto) para graficar.")

        if cBtns[3].button("🔍 Insights de mercado"):
            with left:
                render_ia_html_block(compose_market_text(data), height=520)
            with right:
                # TABLAS (no barras) para Top Marcas / Modelos por CONTEO de OCs/OTs
                def is_marca(n):  return any(k in n for k in ("marca","brand","make","fabricante"))
                def is_modelo(n): return any(k in n for k in ("modelo","model","version","versión","trim"))

                c_top = st.columns(2)

                hM, dfM, catM, gM = best_count_by_category(data, is_marca)
                with c_top[0]:
                    if hM:
                        mostrar_tabla_count(gM, catM, "__count__", "Top Marcas (por OCs/OTs)")
                    else:
                        st.info("Sin columna de **marca**.")

                hMo, dfMo, catMo, gMo = best_count_by_category(data, is_modelo)
                with c_top[1]:
                    if hMo:
                        mostrar_tabla_count(gMo, catMo, "__count__", "Top Modelos (por OCs/OTs)")
                    else:
                        st.info("Sin columna de **modelo**.")

                # Estacionalidad
                found = False
                for h, df in data.items():
                    if df is None or df.empty: continue
                    roles = detect_roles_for_sheet(df, h)
                    date_cols = [c for c,r in roles.items() if r=="date"]
                    money_cols= [c for c,r in roles.items() if r=="money"]
                    if date_cols and money_cols:
                        dt = pd.to_datetime(df[date_cols[0]], errors="coerce")
                        d2 = df.assign(__mm = dt.dt.to_period("M").dt.to_timestamp())
                        d2 = d2.assign(MES = d2["__mm"].dt.strftime("%Y-%m"))
                        mostrar_grafico_linea(d2, "MES", money_cols[0], "Estacionalidad (mes)")
                        found = True
                        break
                if not found:
                    st.info("No encontré (fecha + monto) para estacionalidad.")

        # ------- Responder (junto a la pregunta) -------
        if responder_click and pregunta:
            schema = _build_schema(data)
            plan_c = plan_compute_from_llm(pregunta, schema)
            facts = execute_compute(plan_c, data)

            if not facts.get("ok"):
                with left:
                    st.error(f"No pude calcular con precisión: {facts.get('msg') or 'plan vacío'}. Uso la ruta de análisis clásico.")
                raw = ask_gpt(prompt_consulta_libre(pregunta, schema))
                with left:  render_ia_html_block(raw, height=620)
                with right:
                    ok = False
                    try:
                        plan = plan_from_llm(pregunta, schema)
                        ok = execute_plan(plan, data)
                    except Exception as e:
                        st.error(f"Error ejecutando plan: {e}")
                    if not ok:
                        st.info("Sin visual sugerida para esta consulta.")
                ss.historial.append({"pregunta":pregunta,"respuesta":raw})
            else:
                texto_left = compose_focus_text(facts, pregunta)
                with left:  render_ia_html_block(texto_left, height=520)
                with right:
                    df_res = facts["df_result"]
                    if facts.get("category_col"):
                        try:
                            if facts.get("value_role")=="money" and facts.get("op","sum")!="count":
                                mostrar_grafico_barras_v3(
                                    df_res.rename(columns={facts["category_col"]: "CATEGORIA",
                                                           "VALOR": "VALOR"}),
                                    "CATEGORIA", "VALOR",
                                    f"{facts['op'].upper()} de {facts['value_col']} por {facts['category_col']}"
                                )
                            else:
                                # para conteos o no-moneda, uso tablas si está muy largo
                                st.info("Distribución desbalanceada: muestro tabla para mejor lectura." if len(df_res)>15 else "")
                                df_show = df_res.rename(columns={facts["category_col"]:"Categoría","VALOR":"Valor"}).copy()
                                if facts.get("op","sum")=="count" or facts.get("value_role")!="money":
                                    df_show["Valor"] = df_show["Valor"].apply(lambda v: int(round(float(v))) if pd.notna(v) else 0)
                                else:
                                    df_show["Valor"] = df_show["Valor"].apply(fmt_money)
                                st.markdown("### 📋 Resultado")
                                st.dataframe(df_show, use_container_width=True)
                                st.download_button("⬇️ Descargar tabla (CSV)",
                                                   df_show.to_csv(index=False).encode("utf-8"),
                                                   "resultado.csv","text/csv", key=_unique_key("csv"))
                        except Exception as e:
                            st.error(f"Error graficando: {e}")
                            st.dataframe(df_res, use_container_width=True)
                    else:
                        role = facts.get("value_role","unknown")
                        valtxt = (fmt_money(facts['total']) if role=="money" and facts.get("op","sum")!="count"
                                  else _fmt_number_general(facts['total']))
                        st.metric(f"{facts['op'].upper()} de {facts['value_col']}", valtxt)
                        st.dataframe(df_res, use_container_width=True)
                    st.caption(f"Hoja: {facts['sheet']} • Filas: {facts['rows']}")
                ss.historial.append({"pregunta":pregunta,"respuesta":texto_left})

elif ss.menu_sel == "Historial":
    if ss.historial:
        for i, h in enumerate(ss.historial[-20:], 1):
            st.markdown(f"**Q{i}:** {h['pregunta']}")
            st.markdown(f"**A{i}:**")
            try:
                parsed = json.loads(h["respuesta"]); st.json(parsed)
            except Exception:
                render_ia_html_block(h["respuesta"], height=520)
    else:
        st.info("Aún no hay historial en esta sesión.")

elif ss.menu_sel == "Diagnóstico IA":
    st.markdown("### 🔎 Diagnóstico de la IA (OpenAI)")
    st.caption("Verifica API Key, conexión, prueba mínima de chat y estado de cuota.")
    if st.button("Diagnosticar IA"):
        diag = diagnosticar_openai()
        st.write("**Clave configurada:** ", "✅" if diag["api_key_present"] else "❌")
        st.write("**Organization:** ", "✅" if diag["organization_set"] else "—")
        st.write("**Base URL:** ", "✅" if diag["base_url_set"] else "—")
        st.write("**Listar modelos:** ", "✅" if diag["list_models_ok"] else "❌")
        st.write("**Prueba chat:** ", "✅" if diag["chat_ok"] else "❌")
        if diag["quota_ok"] is True: st.success("Créditos/cuota: OK")
        elif diag["quota_ok"] is False: st.error("❌ Sin créditos/cuota.")
        else: st.info("No se pudo determinar la cuota.")
        if diag["usage_tokens"] is not None: st.caption(f"Tokens: {diag['usage_tokens']}")
        if diag["error"]: st.warning(f"Detalle: {diag['error']}")

