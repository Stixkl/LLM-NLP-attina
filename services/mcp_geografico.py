from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from collections import Counter
from . import config

router = APIRouter(prefix="/analisis", tags=["geografico"])


def get_cardinal_zone(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    if lat is None or lon is None:
        return None
    
    if lat > 66.5:
        return "Norte"
    elif lat < -66.5:
        return "Sur"
    elif lon > 30:
        return "Este"
    elif lon < -30:
        return "Oeste"
    else:
        return "Centro"


class Region(BaseModel):
    region: str
    porcentaje: float
    count: int


class Pais(BaseModel):
    pais: str
    count: int


class MapaData(BaseModel):
    lat: float
    lon: float
    country: Optional[str]
    count: int = 1


class GeograficoRequest(BaseModel):
    conversaciones: list[dict] = Field(default_factory=list)


class GeograficoResponse(BaseModel):
    distribucion: list[Region]
    paises_mas_activos: list[Pais]
    total_ubicaciones: int
    mapa_data: list[MapaData]


@router.post("/geografico", response_model=GeograficoResponse)
async def analisis_geografico(request: GeograficoRequest):
    conversations = request.conversaciones

    if not conversations:
        conversations = config.load_dataset_as_dicts()

    cardinal_counts: dict[str, int] = {}
    country_counts: Counter = Counter()
    coordenadas_unicas: dict[tuple[float, float], int] = {}
    paises_coords: dict[tuple[float, float], str] = {}

    total_con_ubicacion = 0

    for conv in conversations:
        lat_str = conv.get("latitude") or conv.get("lat")
        lon_str = conv.get("longitude") or conv.get("lon")

        lat = float(lat_str) if lat_str and lat_str != "null" else None
        lon = float(lon_str) if lon_str and lon_str != "null" else None

        if lat is not None and lon is not None:
            total_con_ubicacion += 1

            zona = get_cardinal_zone(lat, lon)
            if zona:
                cardinal_counts[zona] = cardinal_counts.get(zona, 0) + 1

            country = conv.get("country")
            if country:
                country_counts[country] += 1

            coord_key = (round(lat, 1), round(lon, 1))
            coordenadas_unicas[coord_key] = coordenadas_unicas.get(coord_key, 0) + 1
            if country:
                paises_coords[coord_key] = country

    all_zones = ["Norte", "Sur", "Este", "Oeste", "Centro"]
    total_zonas = sum(cardinal_counts.values())

    distribucion = []
    for zone in all_zones:
        count = cardinal_counts.get(zone, 0)
        porcentaje = (count / total_zonas * 100) if total_zonas > 0 else 0
        distribucion.append(Region(region=zone, porcentaje=round(porcentaje, 2), count=count))

    paises_top = country_counts.most_common(10)
    paises_mas_activos = [Pais(pais=p, count=c) for p, c in paises_top]

    mapa_data = [
        MapaData(
            lat=lat,
            lon=lon,
            country=paises_coords.get((lat, lon)),
            count=count
        )
        for (lat, lon), count in coordenadas_unicas.items()
    ]

    return GeograficoResponse(
        distribucion=distribucion,
        paises_mas_activos=paises_mas_activos,
        total_ubicaciones=total_con_ubicacion,
        mapa_data=mapa_data
    )