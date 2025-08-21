import pandas as pd
from typing import Dict, Any, Tuple, Optional

def _norm(s: str) -> str:
    return str(s).strip().lower()

def _guess_value_col(df: pd.DataFrame) -> Optional[str]:
    keys = ["monto", "neto", "total", "importe", "facturacion", "ingreso", "venta"]
    for c in df.columns:
        c2 = _norm(c)
        if any(k in c2 for k in keys):
            return c
    return None

def _guess_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        c2 = _norm(c)
        if "fecha" in c2 or "mes" in c2:
            return c
    return None

def _guess_cost_col(df: pd.DataFrame) -> Optional[str]:
    keys = ["costo", "costos", "gasto", "gastos"]
    for c in df.columns:
        if any(k in _norm(c) for k in keys):
            return c
    return None

def _apply_filters(df: pd.DataFrame, filtros: Optional[Tuple[str, str, tuple, str]]) -> pd.DataFrame:
    if not filtros:
        return df
    fecha_col, cliente_col, rango, cliente = filtros
    out = df.copy()
    if fecha_col and fecha_col in out.columns and rango:
        out[fecha_col] = pd.to_datetime(out[fecha_col], errors="coerce")
        ini = pd.to_datetime(rango[0]); fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1)
        out = out[(out[fecha_col] >= ini) & (out[fecha_col] < fin)]
    if cliente and cliente_col and cliente_col in out.columns:
        out = out[out[cliente_col].astype(str).str.contains(cliente, case=False, na=False)]
    return out

def analizar_datos_taller(data: Dict[str, pd.DataFrame], filtros: Optional[Tuple[str,str,tuple,str]]=None) -> Dict[str, Any]:
    """Resumen de negocio multi‑hoja (ingresos, costos, margen, top3 por categoría)."""
    total_ing = 0.0
    total_cost = 0.0
    resumen = {"hojas": {}}

    for hoja, df in data.items():
        if df is None or df.empty:
            continue
        df2 = _apply_filters(df, filtros)

        val = _guess_value_col(df2)
        cost = _guess_cost_col(df2)
        info = {"filas": int(len(df2)), "columnas": list(map(str, df2.columns)), "insights": []}

        # ingresos / costos / margen por hoja
        if val:
            v = pd.to_numeric(df2[val], errors="coerce").fillna(0)
            total_ing += float(v.sum())
        if cost:
            c = pd.to_numeric(df2[cost], errors="coerce").fillna(0)
            total_cost += float(c.sum())

        # top3 categorías típicas
        cats_priority = ["tipo", "cliente", "patente", "estado", "proceso", "vehiculo", "unidad", "mes"]
        candidates = [c for c in df2.columns if df2[c].nunique(dropna=False) <= 30]
        cats_sorted = sorted(candidates, key=lambda c: (0 if any(p in _norm(c) for p in cats_priority) else 1, str(c)))
        if val:
            for cat in cats_sorted[:3]:
                g = (df2.groupby(cat)[val].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
                               .sort_values(ascending=False))
                total = float(g.sum()) if len(g) else 0.0
                top = g.head(3).to_dict()
                conc = float(sum(list(top.values()))/total)*100 if total else 0.0
                info["insights"].append({"categoria": str(cat), "valor": str(val), "top3": top, "concentracion_top3_pct": round(conc,2)})

        resumen["hojas"][hoja] = info

    margen = total_ing - total_cost
    margen_pct = (margen/total_ing*100) if total_ing else 0.0
    resumen.update({
        "ingresos": total_ing,
        "costos": total_cost,
        "margen": margen,
        "margen_pct": round(margen_pct, 2)
    })
    return resumen
