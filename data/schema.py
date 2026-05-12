from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum


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

    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

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
            is_bot=data.get("isBot", "false").lower() == "true" if data.get("isBot") else False,
            influence_score=float(data["influenceScore"]) if data.get("influenceScore") else None,
        )

        location = None
        if data.get("latitude") or data.get("longitude"):
            location = Location(
                country=data.get("country"),
                latitude=float(data["latitude"]) if data.get("latitude") else None,
                longitude=float(data["longitude"]) if data.get("longitude") else None,
            )

        message_type = MessageType.POST
        if data.get("isRetweet", "").lower() == "true":
            message_type = MessageType.RETWEET
        elif data.get("isComment", "").lower() == "true":
            message_type = MessageType.COMMENT

        source = SocialSource.OTHER
        if data.get("sourceName"):
            source_name = data["sourceName"].lower()
            if "twitter" in source_name or "x" in source_name:
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
            id=data["id"],
            text=data.get("text", "") or data.get("caption", "") or "",
            author=author,
            created_at=datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00")) if data.get("createdAt") else datetime.now(),
            message_type=message_type,
            source=source,
            parent_id=data.get("parentId") or None,
            thread_id=data.get("threadId") or None,
            location=location,
            language=data.get("language"),
            sentiment=float(data["sentiment"]) if data.get("sentiment") else None,
            likes=int(data["liked"]) if data.get("liked") else 0,
            is_liked=data.get("liked", "").lower() == "true" if data.get("liked") else False,
            engagement_rate=float(data["engagementRate"]) if data.get("engagementRate") else None,
            tags=data.get("tags", "").split(",") if data.get("tags") else [],
            keywords=data.get("keywords", "").split(",") if data.get("keywords") else [],
            has_media=data.get("hasImageOrVideo", "").lower() == "true" if data.get("hasImageOrVideo") else False,
            is_ad=data.get("isAdvertisement", "").lower() == "true" if data.get("isAdvertisement") else False,
            is_deleted=data.get("isDeleted", "").lower() == "true" if data.get("isDeleted") else False,
            is_archived=data.get("isArchived", "").lower() == "true" if data.get("isArchived") else False,
        )