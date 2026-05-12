# Data Directory

This directory contains all dataset files for the Attina project.

## Structure

```
data/
├── conversations.json       # Main dataset (upload by user)
├── sample_conversations.json # Sample data for testing
└── .gitkeep                  # Placeholder to keep directory in git
```

## Dataset Schema

Each conversation record contains these key fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique message identifier |
| `authorId` | string | Author's unique ID |
| `author` | string | Author's display name |
| `authorCategory` | string | Category (influencer, brand, etc.) |
| `authorURL` | string | Profile URL |
| `text` | string | Message content |
| `createdAt` | ISO 8601 | Publication timestamp |
| `sourceName` | string | Platform (twitter, facebook, etc.) |
| `threadId` | string | Thread/conversation identifier |
| `parentId` | string\|null | Parent message ID (for replies) |
| `isComment` | boolean | Is this a comment? |
| `isRetweet` | boolean | Is this a retweet? |
| `liked` | integer | Like/favorite count |
| `latitude` | float | Geographic latitude |
| `longitude` | float | Geographic longitude |
| `country` | string | Country name |
| `language` | string | Language code (es, en, etc.) |
| `sentiment` | float | Sentiment score (-1 to 1) |
| `influenceScore` | float | Author influence score |
| `isBot` | boolean | Is author a bot? |
| `hasImageOrVideo` | boolean | Has media attachments? |
| `tags` | string | Comma-separated tags |
| `keywords` | string | Comma-separated keywords |
| `isAdvertisement` | boolean | Is this an ad? |
| `isDeleted` | boolean | Is message deleted? |
| `isArchived` | boolean | Is message archived? |

## Loading Data

```python
from data import get_loader

loader = get_loader()
messages = loader.load_json("sample_conversations.json")
stats = loader.get_stats("sample_conversations.json")
```