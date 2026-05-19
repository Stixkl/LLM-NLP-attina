from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers de conversión robustos
# ---------------------------------------------------------------------------

def _parse_datetime(value) -> datetime:
    """Convierte createdAt a datetime manejando múltiples formatos:
    - Unix ms (int/float/string numérico): 1750834500000
    - pandas Timestamp / numpy datetime64
    - String ISO 8601
    """
    if value is None:
        return datetime.now(tz=timezone.utc)
    if hasattr(value, "to_pydatetime"):
        result = value.to_pydatetime()
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    try:
        if pd.isna(value):
            return datetime.now(tz=timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        ms = int(float(str(value)))
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        pass
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc)


def _safe_float(value, default=None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _safe_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes")


def _split_list(val) -> list:
    if not val:
        return []
    return [s.strip() for s in str(val).split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MessageType(str, Enum):
    POST = "post"
    COMMENT = "comment"
    REPLY = "reply"
    RETWEET = "retweet"


class SocialSource(str, Enum):
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Author:
    id: str
    username: str
    url: Optional[str] = None
    category: Optional[str] = None
    is_bot: bool = False
    influence_score: Optional[float] = None


@dataclass
class Location:
    country: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def get_cardinal_direction(self) -> Optional[str]:
        if self.latitude is None or self.longitude is None:
            return None
        lat_zone = "N" if self.latitude > 0 else "S"
        lon_zone = "E" if self.longitude > 0 else "W"
        return f"{lat_zone}{lon_zone}"


@dataclass
class Message:
    id: str
    text: str
    author: Author
    created_at: datetime
    message_type: MessageType
    source: SocialSource

    parent_id: Optional[str] = None
    thread_id: Optional[str] = None

    location: Optional[Location] = None
    language: Optional[str] = None
    sentiment: Optional[float] = None

    likes: int = 0
    is_liked: bool = False
    engagement_rate: Optional[float] = None

    tags: list = field(default_factory=list)
    keywords: list = field(default_factory=list)

    has_media: bool = False
    is_ad: bool = False
    is_deleted: bool = False
    is_archived: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        author_data = data.get("author", {}) or {}
        author = Author(
            id=data.get("authorId", ""),
            username=author_data or data.get("author", data.get("authorId", "")),
            url=data.get("authorURL"),
            category=data.get("authorCategory"),
            is_bot=_safe_bool(data.get("isBot")),
            influence_score=_safe_float(data.get("influenceScore")),
        )

        location = None
        if data.get("latitude") or data.get("longitude"):
            location = Location(
                country=data.get("country"),
                latitude=_safe_float(data.get("latitude")),
                longitude=_safe_float(data.get("longitude")),
            )

        message_type = MessageType.POST
        if _safe_bool(data.get("isRetweet")):
            message_type = MessageType.RETWEET
        elif _safe_bool(data.get("isComment")):
            message_type = MessageType.COMMENT

        source = SocialSource.OTHER
        if data.get("sourceName"):
            source_name = str(data["sourceName"]).lower()
            if "twitter" in source_name or source_name == "x":
                source = SocialSource.TWITTER
            elif "facebook" in source_name:
                source = SocialSource.FACEBOOK
            elif "instagram" in source_name:
                source = SocialSource.INSTAGRAM
            elif "youtube" in source_name:
                source = SocialSource.YOUTUBE
            elif "reddit" in source_name:
                source = SocialSource.REDDIT

        return cls(
            id=str(data["id"]),
            text=data.get("text", "") or data.get("caption", "") or "",
            author=author,
            created_at=_parse_datetime(data.get("createdAt")),
            message_type=message_type,
            source=source,
            parent_id=data.get("parentId") or None,
            thread_id=data.get("threadId") or None,
            location=location,
            language=data.get("language"),
            sentiment=_safe_float(data.get("sentiment")),
            likes=_safe_int(data.get("liked")),
            is_liked=_safe_bool(data.get("liked")),
            engagement_rate=_safe_float(data.get("engagementRate")),
            tags=_split_list(data.get("tags")),
            keywords=_split_list(data.get("keywords")),
            has_media=_safe_bool(data.get("hasImageOrVideo")),
            is_ad=_safe_bool(data.get("isAdvertisement")),
            is_deleted=_safe_bool(data.get("isDeleted")),
            is_archived=_safe_bool(data.get("isArchived")),
        )