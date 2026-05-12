from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Message(BaseModel):
    """
    Modelo que representa un mensaje/post del dataset Attina.
    Los nombres de campo siguen la nomenclatura real del parquet.
    """
    internalId: str = Field(..., description="ID único del mensaje")
    preciseId: Optional[str] = None
    parentId: Optional[str] = Field(None, description="ID del mensaje padre (None si es raíz)")
    parentText: Optional[str] = None
    parentCreatedAt: Optional[str] = None

    author: Optional[str] = None
    authorId: Optional[str] = None
    authorCategory: Optional[str] = None
    authorCategoryId: Optional[str] = None
    authorURL: Optional[str] = None

    caption: Optional[str] = Field(None, description="Texto principal del mensaje")
    description: Optional[str] = None
    metaTitle: Optional[str] = None
    context: Optional[str] = None
    contextTitle: Optional[str] = None
    keywords: Optional[str] = None

    createdAt: Optional[str] = Field(None, description="Timestamp ISO 8601 de publicación")
    socialType: Optional[str] = Field(None, description="Red social: twitter, facebook, tiktok, etc.")
    language: Optional[str] = None
    sentiment: Optional[str] = Field(None, description="Sentimiento: positive, negative, neutral")

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    places: Optional[str] = None

    influenceScore: Optional[float] = None
    engagementRate: Optional[float] = None
    rating: Optional[float] = None
    liked: Optional[bool] = None

    isComment: Optional[bool] = None
    isRetweet: Optional[bool] = None
    isBot: Optional[bool] = None
    isDeleted: Optional[bool] = None
    isArchived: Optional[bool] = None
    isAdvertisement: Optional[bool] = None

    threadId: Optional[str] = None
    sourceId: Optional[str] = None
    sourceName: Optional[str] = None
    link: Optional[str] = None
    host: Optional[str] = None

    def get_text(self) -> str:
        """Retorna el texto principal disponible del mensaje."""
        return self.caption or self.description or self.metaTitle or self.contextTitle or ""

    def get_timestamp(self) -> Optional[datetime]:
        """Parsea el timestamp a datetime si está disponible."""
        raw = self.createdAt
        if not raw:
            return None
        try:
            # Formato: epoch ms (ej: 1750986632000) o ISO string
            if str(raw).isdigit():
                return datetime.fromtimestamp(int(raw) / 1000)
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return None

    def get_parent_timestamp(self) -> Optional[datetime]:
        """Parsea el timestamp del padre si está disponible."""
        raw = self.parentCreatedAt
        if not raw:
            return None
        try:
            if str(raw).isdigit():
                return datetime.fromtimestamp(int(raw) / 1000)
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return None