"""
Base interface for writing plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WritingContext:
    """Context passed to all writing plugins."""

    topic: str
    user: str = "global"
    persona: str | None = None
    tone: str = "neutral"
    audience: str = "general"
    keywords: list[str] = field(default_factory=list)
    style_params: dict[str, Any] = field(default_factory=dict)
    research_enabled: bool = True
    mood: str = "spokój"
    energy: int = 70
    creativity: int = 50


@dataclass
class WritingResult:
    """Result from writing plugin."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, float] = field(default_factory=dict)
    variants: list[str] = field(default_factory=list)


class WritingPlugin(ABC):
    """Base class for all writing plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name/identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass

    @property
    def supported_formats(self) -> list[str]:
        """Supported output formats."""
        return ["md", "html", "txt"]

    @property
    def category(self) -> str:
        """Plugin category (e.g., 'marketplace', 'social', 'creative')."""
        return "general"

    @abstractmethod
    def generate(self, context: WritingContext) -> WritingResult:
        """Generate content based on context."""
        pass

    def validate_context(self, context: WritingContext) -> bool:
        """Validate if context is suitable for this plugin."""
        return True

    def get_default_style(self) -> dict[str, Any]:
        """Get default style parameters for this plugin."""
        return {
            "pace": 0.5,
            "sensory": 0.7,
            "irony": 0.3,
            "slang": 0.2,
            "pathos": 0.4,
            "profanity": 0.0,
        }

    def score_content(self, content: str, context: WritingContext) -> dict[str, float]:
        """Score the generated content (0-100 scale)."""
        return {
            "coherence": 75.0,
            "originality": 70.0,
            "engagement": 65.0,
            "seo_score": 60.0,
            "brand_alignment": 70.0,
        }
