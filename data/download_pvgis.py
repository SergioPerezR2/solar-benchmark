"""
Descarga datos horarios de PVGIS para la ubicación de la estación OMUAQ (UAQ, Querétaro).
Ejecutar desde la raíz del proyecto: python data/download_pvgis.py
"""

import pandas as pd
from pvlib.iotools import get_pvgis_hourly
from pathlib import Path

LAT   = 20.563425
LON   = -100.36943889
YEARS = [(2019, 2020), (2021, 2022)]
OUT   = Path(__file__).parent


def download_block(start, end):
    print(f"  Descargando {start}-{end}...", flush=True)
    data, meta = get_pvgis_hourly(
        latitude=LAT,
        longitude=LON,
        start=start,
        end=end,
        raddatabase="PVGIS-ERA5",
        surface_tilt=0,        # superficie horizontal → POA = GHI
        surface_azimuth=180,
        components=True,
        outputformat="json",
        map_variables=True,
    )
    return data, meta


def main():
    print(f"Ubicación: lat={LAT}, lon={LON}")
    print(f"Base de datos: PVGIS-ERA5 (cobertura: global)")

    frames = []
    for start, end in YEARS:
        df, _ = download_block(start, end)
        frames.append(df)

    data = pd.concat(frames).sort_index()
    data.index.name = "TIMESTAMP"

    # Con superficie horizontal (tilt=0): GHI = poa_direct + poa_sky_diffuse
    data["GHI"] = (data["poa_direct"] + data["poa_sky_diffuse"]).clip(lower=0)

    # Renombrar columnas restantes
    data = data.rename(columns={
        "temp_air":   "TempAir",
        "wind_speed": "WindSpeed",
    })

    # Mantener solo columnas útiles para el benchmark
    keep = ["GHI", "poa_direct", "poa_sky_diffuse", "solar_elevation",
            "TempAir", "WindSpeed"]
    data = data[[c for c in keep if c in data.columns]]

    # Variables temporales (mismas que el dataset local)
    data["hour"]      = data.index.hour
    data["dayofyear"] = data.index.dayofyear
    data["month"]     = data.index.month

    # Guardar
    raw_path = OUT / "pvgis_raw.csv"
    data.to_csv(raw_path)

    print(f"\nGuardado: {raw_path}")
    print(f"Filas   : {len(data):,}")
    print(f"Columnas: {list(data.columns)}")
    print(f"Periodo : {data.index[0]}  →  {data.index[-1]}")
    print(f"GHI  — min: {data['GHI'].min():.1f}  max: {data['GHI'].max():.1f}  "
          f"mean (diurno): {data.loc[data['GHI']>50,'GHI'].mean():.1f} W/m²")
    print(f"Temp — min: {data['TempAir'].min():.1f}  max: {data['TempAir'].max():.1f}  "
          f"mean: {data['TempAir'].mean():.1f} °C")
    print(f"Horas diurnas (GHI>50): {(data['GHI']>50).sum():,} / {len(data):,} "
          f"({100*(data['GHI']>50).mean():.1f}%)")

    print(f"""
--- config a copiar en config.yaml ---
dataset:
  source: pvgis
  latitude: {LAT}
  longitude: {LON}
  database: PVGIS-ERA5
  years: [2019, 2020, 2021, 2022]
  target: GHI
  features: [TempAir, WindSpeed, hour, dayofyear, month]
  ghi_threshold: 50
""")


if __name__ == "__main__":
    main()
