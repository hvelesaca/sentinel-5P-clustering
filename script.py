# Sentinel-5P (NO2) – Guayas / cantones
import ee
import geemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import os
import requests
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ===============================
# PARÁMETROS GENERALES
# ===============================
start_year, end_year = 2020, 2025
scale_m = 7000

# SOLO los estadísticos que indica tu código:
#   - median
#   - percentile con q = 90 / 95 / 99
STAT_TASKS = [
    {"STAT_MODE": "median", "PERCENTILE_Q": None},
    {"STAT_MODE": "percentile", "PERCENTILE_Q": 90},
    {"STAT_MODE": "percentile", "PERCENTILE_Q": 95},
    {"STAT_MODE": "percentile", "PERCENTILE_Q": 99},
]

# Descarga de mapas PNG
DOWNLOAD_MODE = "raster"   # "canton" o "raster"
ADD_BORDERS = True
DIMENSIONS = 240
palette = ["navy", "blue", "cyan", "green", "yellow", "orange", "red"]

# ===============================
# NUEVO: FILTRO MENSUAL (outliers dentro del mismo año)
# ===============================
APPLY_MONTH_STD_FILTER = True
MONTH_FILTER_K = 2.0                 # "fuera de la desviación estándar" => 1σ (ajústalo si quieres 2σ)
MONTH_FILTER_MIN_MONTHS = 9          # si sobreviven menos de 6 meses, no se filtra (fallback)
MONTH_FILTER_REF_COMPOSITE = "median"  # referencia mensual (robusta). Mantener "median" recomendado.

# ===============================
# CONFIGS: NO2
# ===============================
POLLUTANTS = [
    #{
    #    "S5P_COLLECTION": "COPERNICUS/S5P/OFFL/L3_SO2",
    #    "POLLUTANT_BAND": "SO2_column_number_density",
    #    "POLLUTANT_NAME": "SO2",
    #    "POLLUTANT_UNIT": "mol/m²",
    #},
    #{
    #    "S5P_COLLECTION": "COPERNICUS/S5P/OFFL/L3_HCHO",
    #    "POLLUTANT_BAND": "tropospheric_HCHO_column_number_density",
    #    "POLLUTANT_NAME": "HCHO",
    #    "POLLUTANT_UNIT": "mol/m²",
    #},
    #{
    #    "S5P_COLLECTION": "COPERNICUS/S5P/OFFL/L3_CO",
    #    "POLLUTANT_BAND": "CO_column_number_density",
    #    "POLLUTANT_NAME": "CO",
    #    "POLLUTANT_UNIT": "mol/m²",
    #},
    {
        "S5P_COLLECTION": "COPERNICUS/S5P/OFFL/L3_NO2",
        "POLLUTANT_BAND": "tropospheric_NO2_column_number_density",
        "POLLUTANT_NAME": "NO2",
        "POLLUTANT_UNIT": "mol/m²",
    },
]

def save_colorbar_png(out_path, vmin, vmax, palette, title):
    cmap = LinearSegmentedColormap.from_list("gee_palette", palette, N=256)

    fig = plt.figure(figsize=(2.4, 5.0), dpi=220)
    ax = fig.add_axes([0.40, 0.08, 0.22, 0.84])  # [left, bottom, width, height]

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.ax.tick_params(labelsize=8)
    cb.set_label(title, fontsize=9)

    fig.savefig(out_path, bbox_inches="tight", transparent=False)
    plt.close(fig)

# ===============================
# HELPERS (median / percentil)
# ===============================
def build_reducer_and_names(stat_mode, q=None):
    """
    Devuelve:
      reducer: ee.Reducer configurado
      reducer_out: nombre del output del reducer (ej. 'pollutant_stat')
      img_band: nombre de banda de la imagen anual (ej. 'pollutant_stat')
      label_suffix: texto para columnas/títulos (ej. 'median' o 'p95')
    """
    if stat_mode == "median":
        reducer = ee.Reducer.median().setOutputs(["pollutant_stat"])
        label_suffix = "median"
    elif stat_mode == "percentile":
        if q is None:
            raise ValueError("Si STAT_MODE='percentile' debes definir PERCENTILE_Q (90/95/99).")
        q = int(q)
        reducer = ee.Reducer.percentile([q]).setOutputs(["pollutant_stat"])
        label_suffix = f"p{q}"
    else:
        raise ValueError("STAT_MODE inválido. Usa 'median' o 'percentile'.")

    reducer_out = "pollutant_stat"
    img_band = "pollutant_stat"
    return reducer, reducer_out, img_band, label_suffix

def annual_image(ic_year, stat_mode, q=None):
    if stat_mode == "median":
        return ic_year.median().rename("pollutant_stat")

    elif stat_mode == "percentile":
        q = int(q)
        p = ic_year.reduce(ee.Reducer.percentile([q])).rename("pollutant_stat")
        # fallback: rellena huecos del percentil con la mediana del mismo año
        med = ic_year.median().rename("pollutant_stat")
        return p.unmask(med)

    else:
        raise ValueError("STAT_MODE inválido. Usa 'median' o 'percentile'.")

def annual_image2(ic_year, stat_mode, q=None):
    """Imagen anual (1 banda) consistente con STAT_MODE."""
    if stat_mode == "median":
        return ic_year.median().rename("pollutant_stat")
    elif stat_mode == "percentile":
        q = int(q)
        return ic_year.reduce(ee.Reducer.percentile([q])).rename("pollutant_stat")
    else:
        raise ValueError("STAT_MODE inválido. Usa 'median' o 'percentile'.")

def set_sci_y(ax):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# ===============================
# NUEVO: HELPERS filtro mensual por año
# ===============================
def _monthly_composite(ic_m, ref_composite="median"):
    # Por ahora mantenemos "median" para referencia mensual robusta
    if ref_composite == "median":
        return ic_m.median()
    raise ValueError("MONTH_FILTER_REF_COMPOSITE inválido. Usa 'median'.")

def compute_months_keep_by_year(*, s5p_ic, pollutant_band, bounds, years, scale_m,
                                k_sigma=1.0, min_months_keep=6, ref_composite="median",
                                verbose=True):
    """
    Devuelve dict: {year: {"months_keep":[...], "months_avail":[...], "median":..., "std":...}}
    Criterio: se eliminan MESES del año cuyos valores mensuales (promedio provincial del compuesto mensual)
    quedan fuera de mediana ± k_sigma*std.
    """
    out = {}
    for y in years:
        start = ee.Date.fromYMD(int(y), 1, 1)
        end = start.advance(1, "year")
        ic_y = s5p_ic.filterDate(start, end)

        # Conteo rápido
        n_raw = ic_y.size().getInfo()
        if n_raw == 0:
            out[int(y)] = {"months_keep": [], "months_avail": [], "median": None, "std": None, "n_raw": 0}
            if verbose:
                print(f"   >> Filtro mensual {y}: sin imágenes (n_raw=0).")
            continue

        months_avail, vals = [], []
        for m in range(1, 13):
            ic_m = ic_y.filter(ee.Filter.calendarRange(m, m, "month"))
            n_m = ic_m.size().getInfo()
            if n_m == 0:
                continue

            img_m = _monthly_composite(ic_m, ref_composite=ref_composite)
            d = img_m.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=bounds,
                scale=scale_m,
                maxPixels=1e13,
                bestEffort=True,
                tileScale=4,
            ).getInfo()

            v = d.get(pollutant_band)
            if v is None:
                continue

            months_avail.append(int(m))
            vals.append(float(v))

        if len(vals) < 2:
            # Muy pocos meses con valor => no se filtra
            out[int(y)] = {
                "months_keep": months_avail,
                "months_avail": months_avail,
                "median": (float(vals[0]) if len(vals) == 1 else None),
                "std": 0.0,
                "n_raw": n_raw
            }
            if verbose:
                print(f"   >> Filtro mensual {y}: meses_avail={months_avail} (pocos datos) => sin filtrado.")
            continue

        med = float(np.median(vals))
        sd = float(np.std(vals))  # ddof=0

        if (sd == 0.0) or (not np.isfinite(sd)):
            months_keep = months_avail[:]
        else:
            months_keep = [m for m, v in zip(months_avail, vals) if abs(v - med) <= (k_sigma * sd)]

        # Fallback si el filtro deja muy pocos meses
        if (len(months_keep) < min_months_keep) and (len(months_avail) >= min_months_keep):
            months_keep = months_avail[:]

        out[int(y)] = {
            "months_keep": months_keep,
            "months_avail": months_avail,
            "median": med,
            "std": sd,
            "n_raw": n_raw
        }

        if verbose:
            print(
                f"   >> Filtro mensual {y}: avail={months_avail} keep={months_keep} "
                f"(med={med:.4g}, std={sd:.4g}, K={k_sigma})"
            )

    return out

def apply_month_filter(ic_y, months_keep):
    """
    Filtra ImageCollection a sólo meses en months_keep (lista de ints 1..12).
    Si months_keep está vacío, devuelve ic_y SIN filtrar (fallback seguro).
    """
    if not months_keep:
        return ic_y

    months_keep = [int(m) for m in months_keep]

    def set_month(img):
        m = ee.Date(img.get("system:time_start")).get("month").toInt()
        return img.set("month", m)

    return (
        ic_y.map(set_month)
            .filter(ee.Filter.inList("month", ee.List(months_keep)))
    )

# ===============================
# 0) Inicializar Google Earth Engine (1 sola vez)
# ===============================
print("\n >> Inicializando Earth Engine...")
try:
    #ee.Authenticate()
    ee.Initialize(project="gee-ecuador")
    print(" >> ✅ Conexión establecida sin autenticación.")
except Exception:
    print(" >> ⚠ No estaba autenticado. Iniciando autenticación...")
    ee.Authenticate()
    ee.Initialize(project="gee-ecuador")
    print(" >> ✅ Autenticación completada.")
print(ee.String(" >> Hello from Earth Engine!").getInfo())

# ===============================
# 1) Cargar geometría de la provincia del Guayas (1 sola vez)
# ===============================
print("\n >> Cargando límites de Guayas...")
admin1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
guayas_fc = (
    admin1.filter(ee.Filter.eq("ADM0_NAME", "Ecuador"))
          .filter(ee.Filter.eq("ADM1_NAME", "Guayas"))
)
if guayas_fc.size().getInfo() == 0:
    guayas_fc = (
        admin1.filter(ee.Filter.eq("ADM0_NAME", "Ecuador"))
              .filter(ee.Filter.stringContains("ADM1_NAME", "Guayas"))
    )
bounds = guayas_fc.geometry().dissolve()
print(f" >> ✅ Provincia cargada. Features: {guayas_fc.size().getInfo()}")

# ===============================
# 2) Cargar cantones del Guayas (1 sola vez)
# ===============================
print("\n >> Cargando cantones...")
admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
cantones_guayas = (
    admin2.filter(ee.Filter.eq("ADM0_NAME", "Ecuador"))
          .filter(ee.Filter.eq("ADM1_NAME", "Guayas"))
)
print(f" >> ✅ Cantones cargados. Total: {cantones_guayas.size().getInfo()}")

# Para el caso "sin datos", necesitamos lista de nombres (una vez)
CANTON_NAMES = cantones_guayas.aggregate_array("ADM2_NAME").getInfo()

# ===============================
# FUNCIÓN: ejecuta TODO para 1 contaminante + 1 estadístico
# ===============================
def run_pollutant_one_stat(*, s5p_ic, POLLUTANT_NAME, POLLUTANT_UNIT, POLLUTANT_BAND,
                          STAT_MODE, PERCENTILE_Q, months_filter_by_year):
    reducer, reducer_out, img_band, STAT_LABEL = build_reducer_and_names(STAT_MODE, PERCENTILE_Q)
    value_col = f"{POLLUTANT_NAME}_{STAT_LABEL}_{POLLUTANT_UNIT}"

    print(f"\n   >> Estadístico: {STAT_LABEL}")

    # ===============================
    # 4) Prueba de reducción por cantón para un año
    # ===============================
    print("   >> Reduciendo imagen por cantón para un año (prueba)...")
    test_year = 2020

    ic_test_raw = s5p_ic.filterDate(f"{test_year}-01-01", f"{test_year+1}-01-01")
    if APPLY_MONTH_STD_FILTER:
        months_keep = months_filter_by_year.get(test_year, {}).get("months_keep", [])
        ic_test = apply_month_filter(ic_test_raw, months_keep)
    else:
        ic_test = ic_test_raw

    annual_test_img = annual_image(ic_test, STAT_MODE, PERCENTILE_Q)

    fc_test = annual_test_img.reduceRegions(
        collection=cantones_guayas,
        reducer=reducer,
        scale=scale_m,
        tileScale=4,
    )

    first_dict = fc_test.first().toDictionary().getInfo()
    print(f"   >> Primer feature: {first_dict}")
    print(f"   >> Valor {STAT_LABEL} {POLLUTANT_NAME}: {first_dict.get(reducer_out)}")

    # ===============================
    # 5) Generar CSV anual por cantón (ahora con filtro mensual por año)
    # ===============================
    print(f"   >> Calculando {POLLUTANT_NAME} anual por cantón ({STAT_LABEL})...")

    rows = []
    for y in range(start_year, end_year + 1):
        start_y = ee.Date.fromYMD(y, 1, 1)
        end_y = start_y.advance(1, "year")

        ic_y_raw = s5p_ic.filterDate(start_y, end_y)
        n_imgs_raw = int(ic_y_raw.size().getInfo())

        if n_imgs_raw == 0:
            for c in CANTON_NAMES:
                rows.append({
                    "canton": c,
                    "year": y,
                    "n_images": n_imgs_raw,
                    value_col: None
                })
            continue

        if APPLY_MONTH_STD_FILTER:
            months_keep = months_filter_by_year.get(y, {}).get("months_keep", [])
            ic_y = apply_month_filter(ic_y_raw, months_keep)
            n_imgs_used = int(ic_y.size().getInfo())
            # Fallback fuerte si por alguna razón el filtrado dejó 0 imágenes
            if n_imgs_used == 0:
                ic_y = ic_y_raw
        else:
            ic_y = ic_y_raw

        annual_img = annual_image(ic_y, STAT_MODE, PERCENTILE_Q)

        fc_y = annual_img.reduceRegions(
            collection=cantones_guayas,
            reducer=reducer,
            scale=scale_m,
            tileScale=4,
        )

        # set propiedades consistentes
        fc_y = fc_y.map(lambda f: f.set({
            "canton": f.get("ADM2_NAME"),
            "year": y,
            "n_images": n_imgs_raw,
            "value": f.get(reducer_out)
        }))

        info_y = fc_y.getInfo()
        for ft in info_y["features"]:
            p = ft["properties"]
            rows.append({
                "canton": p.get("canton"),
                "year": p.get("year"),
                "n_images": p.get("n_images"),
                value_col: p.get("value")
            })

    df_cantones = (
        pd.DataFrame(rows)
          .sort_values(["canton", "year"])
          .reset_index(drop=True)
    )
    df_cantones[value_col] = pd.to_numeric(df_cantones[value_col], errors="coerce")

    csv_filename = f"{POLLUTANT_NAME.lower()}_cantones_guayas_{start_year}_{end_year}_{STAT_LABEL}.csv"
    df_cantones.to_csv(csv_filename, index=False)
    print(f"   >> ✅ CSV generado: {csv_filename}")

    # ===============================
    # 6) Visualización
    # ===============================
    print(f"   >> Generando gráficos de {POLLUTANT_NAME} ({STAT_LABEL})...")

    df_plot = df_cantones.dropna(subset=["year", "canton"]).copy()
    df_plot["year"] = df_plot["year"].astype(int)

    sns.set_theme(style="whitegrid")
    sns.set_palette("tab10")

    TOP_N = 10
    top_cantones = (
        df_plot.groupby("canton")[value_col]
               .mean()
               .sort_values(ascending=False)
               .head(TOP_N)
               .index
               .tolist()
    )
    df_top = df_plot[df_plot["canton"].isin(top_cantones)].copy()
    canton_order_top = (
        df_top.groupby("canton")[value_col]
              .mean()
              .sort_values(ascending=False)
              .index
              .tolist()
    )

    # -------- LINE PLOT --------
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_top,
        x="year",
        y=value_col,
        hue="canton",
        hue_order=canton_order_top,
        marker="o"
    )

    prov_stat = df_plot.groupby("year")[value_col].mean().reset_index()
    sns.lineplot(
        data=prov_stat,
        x="year",
        y=value_col,
        color="black",
        linewidth=3,
        marker="o",
        label="Total Mean"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(f"{POLLUTANT_NAME} ({POLLUTANT_UNIT})")
    set_sci_y(ax)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    plt.close()

    # -------- BOXPLOT --------
    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(
        data=df_top,
        x="canton",
        y=value_col,
        order=canton_order_top,
        palette="Dark2"
    )
    sns.stripplot(
        data=df_top,
        x="canton",
        y=value_col,
        order=canton_order_top,
        color="black",
        size=3,
        alpha=0.5
    )
    ax.set_xlabel("Canton")
    ax.set_ylabel(f"{POLLUTANT_NAME} ({POLLUTANT_UNIT})")
    plt.xticks(rotation=45, ha="right")
    set_sci_y(ax)
    plt.tight_layout()
    plt.show()
    plt.close()

    # -------- HEATMAP --------
    pivot_all = df_plot.pivot_table(
        index="canton",
        columns="year",
        values=value_col,
        aggfunc="mean"
    )
    pivot_all = pivot_all.loc[pivot_all.mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(10, max(7, 0.32 * len(pivot_all))))
    ax = sns.heatmap(
        pivot_all,
        cmap="RdYlGn_r",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": f"{POLLUTANT_NAME} ({POLLUTANT_UNIT})"}
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Canton")

    cbar = ax.collections[0].colorbar
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.tight_layout()
    plt.show()
    plt.close()

    # ===============================
    # 7) Descargar mapas PNG por año (SIN geemap.Map)
    # ===============================
    OUT_DIR = f"png_maps_{POLLUTANT_NAME.lower()}_{STAT_LABEL}_{start_year}_{end_year}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ===============================
    # Escala GLOBAL fija (p2/p98) para TODO el periodo
    # NUEVO: la calculamos sobre la mediana de las IMÁGENES ANUALES YA FILTRADAS
    # (así evita escalas infladas por meses outlier)
    # ===============================
    annual_imgs = []
    for y in range(start_year, end_year + 1):
        start_y = ee.Date.fromYMD(y, 1, 1)
        end_y = start_y.advance(1, "year")
        ic_y_raw = s5p_ic.filterDate(start_y, end_y)

        if APPLY_MONTH_STD_FILTER:
            months_keep = months_filter_by_year.get(y, {}).get("months_keep", [])
            ic_y = apply_month_filter(ic_y_raw, months_keep)
            if int(ic_y.size().getInfo()) == 0:
                ic_y = ic_y_raw
        else:
            ic_y = ic_y_raw

        if int(ic_y.size().getInfo()) == 0:
            continue

        annual_imgs.append(annual_image(ic_y, STAT_MODE, PERCENTILE_Q).clip(bounds))

    if len(annual_imgs) == 0:
        raise RuntimeError(f"No hay imágenes anuales para escalar {POLLUTANT_NAME} {STAT_LABEL}.")

    ic_annual = ee.ImageCollection.fromImages(annual_imgs)
    img_all = ic_annual.median().rename("pollutant_stat").clip(bounds)

    # Si estás en modo "canton", calculamos la escala global sobre el raster cantonizado
    if DOWNLOAD_MODE == "canton":
        fc_all = img_all.reduceRegions(
            collection=cantones_guayas,
            reducer=reducer,
            scale=scale_m,
            tileScale=4
        ).map(lambda f: f.set("value", f.get(reducer_out)))

        img_all_for_scale = fc_all.reduceToImage(
            properties=["value"],
            reducer=ee.Reducer.first()
        ).rename("pollutant_stat").clip(bounds)
    else:
        img_all_for_scale = img_all

    pct_global = img_all_for_scale.reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]),
        geometry=bounds,
       scale=scale_m,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    vmin_global = pct_global.get("pollutant_stat_p2")
    vmax_global = pct_global.get("pollutant_stat_p98")

    if vmin_global is None or vmax_global is None:
        # Fallback a min/max si percentiles no salen
        mm_global = img_all_for_scale.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=bounds,
            scale=scale_m,
            maxPixels=1e13,
            bestEffort=True
        ).getInfo()
        vmin_global = mm_global.get("pollutant_stat_min")
        vmax_global = mm_global.get("pollutant_stat_max")

    if vmax_global is None or vmin_global is None:
        raise RuntimeError(
            f"No se pudo calcular escala global para {POLLUTANT_NAME} {STAT_LABEL}. "
            f"Revisa enmascarado/datos o aumenta scale_m."
        )

    if vmax_global <= vmin_global:
        vmax_global = vmin_global + 1e-12

    print(f"   >> Escala GLOBAL fija {POLLUTANT_NAME} {STAT_LABEL}: vmin={vmin_global} vmax={vmax_global}")

    # Guardar 1 sola imagen de leyenda global
    legend_global_name = f"{POLLUTANT_NAME.lower()}_{STAT_LABEL}_{DOWNLOAD_MODE}_GLOBAL_legend.png"
    legend_global_path = os.path.join(OUT_DIR, legend_global_name)
    legend_title = f"{POLLUTANT_NAME} ({POLLUTANT_UNIT})\n{STAT_LABEL}  p2–p98  GLOBAL {start_year}-{end_year}"
    save_colorbar_png(legend_global_path, vmin_global, vmax_global, palette, legend_title)
    print(f"   ✅ Leyenda GLOBAL guardada -> {legend_global_path}")

    # ===============================
    # Ahora generamos mapas por año usando SIEMPRE la misma escala global
    # (y usando colecciones anuales filtradas por meses)
    # ===============================
    for y in range(start_year, end_year + 1):
        start_y = ee.Date.fromYMD(y, 1, 1)
        end_y = start_y.advance(1, "year")

        ic_y_raw = s5p_ic.filterDate(start_y, end_y)

        if APPLY_MONTH_STD_FILTER:
            months_keep = months_filter_by_year.get(y, {}).get("months_keep", [])
            ic_y = apply_month_filter(ic_y_raw, months_keep)
            if int(ic_y.size().getInfo()) == 0:
                ic_y = ic_y_raw
        else:
            ic_y = ic_y_raw

        annual_img_y = annual_image(ic_y, STAT_MODE, PERCENTILE_Q).clip(bounds)

        if DOWNLOAD_MODE == "raster":
            base_img = annual_img_y

        elif DOWNLOAD_MODE == "canton":
            fc_y = annual_img_y.reduceRegions(
                collection=cantones_guayas,
                reducer=reducer,
                scale=scale_m,
                tileScale=4
            ).map(lambda f: f.set("value", f.get(reducer_out)))

            base_img = fc_y.reduceToImage(
                properties=["value"],
                reducer=ee.Reducer.first()
            ).rename("pollutant_stat").clip(bounds)

        else:
            raise ValueError("DOWNLOAD_MODE inválido. Usa 'canton' o 'raster'.")

        rgb = base_img.visualize(min=vmin_global, max=vmax_global, palette=palette)

        if ADD_BORDERS:
            borders = ee.Image().byte().paint(
                featureCollection=cantones_guayas, color=1, width=1
            ).visualize(palette=["000000"], opacity=0.85)
            rgb = rgb.blend(borders)

            prov_border = ee.Image().byte().paint(
                featureCollection=guayas_fc, color=1, width=2
            ).visualize(palette=["000000"], opacity=1.0)
            rgb = rgb.blend(prov_border)

        thumb_params = {"region": bounds, "dimensions": DIMENSIONS, "format": "png"}
        url = rgb.getThumbURL(thumb_params)

        fname = f"{POLLUTANT_NAME.lower()}_{STAT_LABEL}_{DOWNLOAD_MODE}_{y}.png"
        out_path = os.path.join(OUT_DIR, fname)

        r = requests.get(url, timeout=1200)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)

        print(f"   ✅ {POLLUTANT_NAME} {STAT_LABEL} {y}: guardado -> {out_path}")


# ===============================
# FUNCIÓN: ejecuta TODOS los estadísticos (median + p90/p95/p99) para 1 contaminante
# ===============================
def run_pollutant(cfg):
    POLLUTANT_NAME = cfg["POLLUTANT_NAME"]
    POLLUTANT_UNIT = cfg["POLLUTANT_UNIT"]
    S5P_COLLECTION = cfg["S5P_COLLECTION"]
    POLLUTANT_BAND = cfg["POLLUTANT_BAND"]

    print(f"\n\n==============================")
    print(f"==>> INICIANDO CONTAMINANTE: {POLLUTANT_NAME}")
    print(f"==============================")

    print(f"\n >> Preparando colección Sentinel-5P {POLLUTANT_NAME}...")
    s5p_ic = (
        ee.ImageCollection(S5P_COLLECTION)
          .filterBounds(bounds)
          .select(POLLUTANT_BAND)
    )
    print(f" >> ✅ Total imágenes: {s5p_ic.size().getInfo()}")

    # ===============================
    # NUEVO: calcular (una vez) qué meses conservar por año
    # ===============================
    years = list(range(start_year, end_year + 1))
    if APPLY_MONTH_STD_FILTER:
        print(f"\n >> Filtro mensual ACTIVADO: K={MONTH_FILTER_K}, min_months_keep={MONTH_FILTER_MIN_MONTHS}, ref={MONTH_FILTER_REF_COMPOSITE}")
        months_filter_by_year = compute_months_keep_by_year(
            s5p_ic=s5p_ic,
            pollutant_band=POLLUTANT_BAND,
            bounds=bounds,
            years=years,
            scale_m=scale_m,
            k_sigma=MONTH_FILTER_K,
            min_months_keep=MONTH_FILTER_MIN_MONTHS,
            ref_composite=MONTH_FILTER_REF_COMPOSITE,
            verbose=True
        )
    else:
        months_filter_by_year = {y: {"months_keep": []} for y in years}

    # Corre todos los estadísticos para este contaminante (uno por uno)
    for task in STAT_TASKS:
        run_pollutant_one_stat(
            s5p_ic=s5p_ic,
            POLLUTANT_NAME=POLLUTANT_NAME,
            POLLUTANT_UNIT=POLLUTANT_UNIT,
            POLLUTANT_BAND=POLLUTANT_BAND,
            STAT_MODE=task["STAT_MODE"],
            PERCENTILE_Q=task["PERCENTILE_Q"],
            months_filter_by_year=months_filter_by_year
        )


# ===============================
# EJECUCIÓN SECUENCIAL
# ===============================
for cfg in POLLUTANTS:
    run_pollutant(cfg)