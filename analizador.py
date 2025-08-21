import pandas as pd
from typing import Dict, Any, Optional

def _norm(s: str) -> str:
    return str(s).strip().lower()

# Pistas genéricas (funciona con planillas distintas)
ING_HINTS = ["monto", "neto", "total", "importe", "facturacion", "ingreso", "venta", "principal"]
COST_HINTS = [
    "costo", "costos", "gasto", "gastos", "insumo", "insumos",
    "repuesto", "repuestos", "pintura", "material", "materiales",
    "mano de obra", "mo", "imposicion", "imposiciones",
    "arriendo", "admin", "administracion", "equipo de planta", "generales",
    "financiero", "financieros", "interes", "intereses"
]

def _find_numeric_cols(df: pd.DataFrame, keywords):
    cols = []
    for c in df.columns:
        c2 = _norm(c)
        if any(k in c2 for k in keywords):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                cols.append(c)
    return cols

def _count_services(df: pd.DataFrame) -> int:
    # heurística por identificadores frecuentes
    keys = ["patente", "orden", "oc", "folio", "documento", "presupuesto", "id", "nro", "número", "num"]
    for k in keys:
        for c in df.columns:
            if k in _norm(c):
                return int(df[c].nunique(dropna=True))
    # fallback: filas con ingreso > 0
    ing_cols = _find_numeric_cols(df, ING_HINTS)
    if ing_cols:
        s = pd.to_numeric(df[ing_cols[0]], errors="coerce")
        return int((s > 0).sum())
    return 0

def _lead_time_days(df: pd.DataFrame):
    # si existen 2 fechas típicas (ingreso/salida), reporta mediana en días
    start_keys = ["fecha ingreso", "ingreso", "recepcion", "entrada"]
    end_keys   = ["fecha salida", "salida", "entrega", "egreso", "termino", "término"]
    start_col = end_col = None
    for c in df.columns:
        cc = _norm(c)
        if not start_col and any(k in cc for k in start_keys): start_col = c
        if not end_col and any(k in cc for k in end_keys):     end_col = c
    if start_col and end_col:
        s = pd.to_datetime(df[start_col], errors="coerce")
        e = pd.to_datetime(df[end_col], errors="coerce")
        d = (e - s).dt.days
        d = d[(d.notna()) & (d >= 0) & (d < 365)]
        if len(d) >= 3:
            return float(d.median())
    return None

def _apply_client_filter(df: pd.DataFrame, client_substr: str) -> pd.DataFrame:
    if not client_substr:
        return df
    # detecta columna cliente si existe
    cliente_col = None
    for c in df.columns:
        if "cliente" in _norm(c):
            cliente_col = c
            break
    if cliente_col:
        out = df.copy()
        return out[out[cliente_col].astype(str).str.contains(client_substr, case=False, na=False)]
    return df

def analizar_datos_taller(data: Dict[str, pd.DataFrame], cliente_contiene: str = "") -> Dict[str, Any]:
    """
    KPIs genéricos multi-hoja (sin rango de fechas):
    - ingresos: suma de TODAS las columnas numéricas con hints de ingreso (en todas las hojas)
    - costos: suma de TODAS las columnas numéricas con hints de costo (en todas las hojas)
    - margen, margen_pct
    - servicios: conteo (por patente/orden/folio/... o filas con ingreso>0)
    - ticket_promedio = ingresos / servicios
    - conversion_pct (si existe una columna 'estado' con Ganado/Perdido/Enviado)
    - lead_time_mediano_dias (si hay fechas de ingreso/salida)
    """
    total_ing = 0.0
    total_cost = 0.0
    total_services = 0
    conversion = None
    lead_time = None

    hojas = {}

    for hoja, df in data.items():
        if df is None or df.empty:
            continue

        df2 = _apply_client_filter(df, cliente_contiene)

        ing_cols = _find_numeric_cols(df2, ING_HINTS)
        cost_cols = _find_numeric_cols(df2, COST_HINTS)

        ing_sum = float(pd.DataFrame({c: pd.to_numeric(df2[c], errors="coerce") for c in ing_cols}).sum().sum()) if ing_cols else 0.0
        cost_sum = float(pd.DataFrame({c: pd.to_numeric(df2[c], errors="coerce") for c in cost_cols}).sum().sum()) if cost_cols else 0.0
        total_ing += ing_sum
        total_cost += cost_sum

        total_services += _count_services(df2)

        # conversión si hay 'estado'
        est_col = None
        for c in df2.columns:
            if "estado" in _norm(c) or "resultado" in _norm(c):
                est_col = c; break
        if est_col is not None:
            est = df2[est_col].astype(str).str.lower()
            gan = int(est.str.contains("ganad", na=False).sum())
            per = int(est.str.contains("perdid", na=False).sum())
            env = int(est.str.contains("enviad", na=False).sum())
            den = gan + per + env
            if den > 0:
                conversion = round(gan / den * 100.0, 2)

        # lead time (si hay fechas)
        lt = _lead_time_days(df2)
        if lt is not None:
            lead_time = lt

        hojas[hoja] = {
            "filas": int(len(df2)),
            "ing_cols": ing_cols,
            "cost_cols": cost_cols,
            "ingresos_hoja": ing_sum,
            "costos_hoja": cost_sum
        }

    margen = total_ing - total_cost
    margen_pct = (margen / total_ing * 100.0) if total_ing else 0.0
    ticket = (total_ing / total_services) if total_services else None

    return {
        "ingresos": total_ing,
        "costos": total_cost,
        "margen": margen,
        "margen_pct": round(margen_pct, 2),
        "servicios": total_services,
        "ticket_promedio": ticket,
        "conversion_pct": conversion,
        "lead_time_mediano_dias": lead_time,
        "hojas": hojas
    }

