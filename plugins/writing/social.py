"""
Social media writing plugin for posts, threads, and viral content.
"""

from __future__ import annotations

import re
from typing import Any

from .base import WritingContext, WritingPlugin, WritingResult


class SocialMediaPlugin(WritingPlugin):
    """Plugin for generating social media posts and threads."""

    @property
    def name(self) -> str:
        return "social"

    @property
    def description(self) -> str:
        return "Generates engaging social media posts and threads"

    @property
    def category(self) -> str:
        return "social"

    def validate_context(self, context: WritingContext) -> bool:
        """Social media plugin accepts any topic for social content."""
        return bool(context.topic and len(context.topic.strip()) > 3)

    def get_default_style(self) -> dict[str, Any]:
        """Social media style: engaging, casual, visual."""
        return {
            "pace": 0.7,
            "sensory": 0.8,
            "irony": 0.4,
            "slang": 0.6,
            "pathos": 0.7,
            "profanity": 0.1,
        }

    def generate(self, context: WritingContext) -> WritingResult:
        """Generate social media content."""
        platform = self._detect_platform(context.topic)
        content_type = self._detect_content_type(context.topic)

        if content_type == "thread":
            content = self._generate_thread(context, platform)
        else:
            content = self._generate_post(context, platform)

        # Score the content
        scoring = self.score_content(content, context)

        # Metadata
        metadata = {
            "platform": platform,
            "content_type": content_type,
            "character_count": len(content),
            "estimated_engagement": self._estimate_engagement(content),
            "hashtag_count": len(re.findall(r"#\w+", content)),
            "call_to_action": (
                "Yes"
                if any(
                    cta in content.lower()
                    for cta in ["komentarz", "udostępnij", "obserwuj", "like"]
                )
                else "No"
            ),
            "score": sum(scoring.values()) / len(scoring),
        }

        return WritingResult(content=content, metadata=metadata, scoring=scoring)

    def _detect_platform(self, topic: str) -> str:
        """Detect target platform from topic."""
        topic_lower = topic.lower()
        if any(p in topic_lower for p in ["twitter", "x.com", "tweet"]):
            return "twitter"
        elif any(p in topic_lower for p in ["linkedin", "professional"]):
            return "linkedin"
        elif any(p in topic_lower for p in ["instagram", "insta", "ig"]):
            return "instagram"
        elif any(p in topic_lower for p in ["facebook", "fb"]):
            return "facebook"
        return "general"

    def _detect_content_type(self, topic: str) -> str:
        """Detect content type (post, thread, story)."""
        topic_lower = topic.lower()
        if any(t in topic_lower for t in ["thread", "wątek", "seria"]):
            return "thread"
        elif any(t in topic_lower for t in ["story", "historie"]):
            return "story"
        return "post"

    def _generate_post(self, context: WritingContext, platform: str) -> str:
        """Generate a single social media post."""
        hook = self._generate_hook(context)
        main_content = self._generate_main_content(context, platform)
        cta = self._generate_cta(platform)
        hashtags = self._generate_hashtags(context, platform)

        content = f"{hook}\n\n{main_content}\n\n{cta}\n\n{hashtags}"
        return content.strip()

    def _generate_thread(self, context: WritingContext, platform: str) -> str:
        """Generate a multi-post thread."""
        thread_points = [self._generate_thread_point(context, i, 5) for i in range(1, 6)]

        thread = []
        for i, point in enumerate(thread_points, 1):
            prefix = f"{i}/🧵 " if i == 1 else f"{i}/ "
            thread.append(f"{prefix}{point}")

        hashtags = self._generate_hashtags(context, platform)
        thread.append(f"\n{hashtags}")

        return "\n\n".join(thread)

    def _generate_hook(self, context: WritingContext) -> str:
        """Generate an attention-grabbing hook."""
        mood = context.mood or "neutral"

        if mood == "energiczny":
            hooks = [
                f"🔥 To zmieni twoje podejście do: {context.topic}",
                f"💯 Właśnie odkryłem coś ważnego o: {context.topic}",
                f"⚡ Game changer w temacie: {context.topic}",
            ]
        elif mood == "spokojny":
            hooks = [
                "📚 Nauczyłem się czegoś ważnego:",
                f"💭 Przemyślenie na temat: {context.topic}",
                "🤔 Interesujące spostrzeżenie:",
            ]
        else:
            hooks = [
                "📚 Nauczyłem się czegoś ważnego:",
                f"🎯 Dziś o tym myślę: {context.topic}",
                f"💡 Ważna rzecz na temat: {context.topic}",
            ]

        import random

        return random.choice(hooks)

    def _generate_main_content(self, context: WritingContext, platform: str) -> str:
        """Generate main post content."""
        content = f"O tym właśnie myślę: {context.topic}"

        if platform == "linkedin":
            content += (
                "\n\nZ mojego doświadczenia wynika, że to właśnie "
                "taki approach przynosi najlepsze rezultaty."
            )
        elif platform == "twitter":
            content += "\n\nKrótko mówiąc: warto nad tym się zastanowić."

        return content

    def _generate_cta(self, platform: str) -> str:
        """Generate call-to-action."""
        ctas = {
            "twitter": "💬 Co myślisz? RT jeśli się zgadzasz!",
            "linkedin": "👉 Co myślisz o tym podejściu? "
            "Podziel się swoją opinią w komentarzach!",
            "instagram": "💝 Zapisz ten post jeśli był przydatny!",
            "facebook": "👍 Daj znać w komentarzu, " "czy masz podobne doświadczenia!",
            "general": "💭 Podziel się swoimi przemyśleniami w komentarzach!",
        }
        return ctas.get(platform, ctas["general"])

    def _generate_hashtags(self, context: WritingContext, platform: str) -> str:
        """Generate relevant hashtags."""
        topic_words = re.findall(r"\b\w{3,}\b", context.topic.lower())
        base_tags = topic_words[:3]

        platform_hashtags = {
            "twitter": ["content", "thread", "thoughts"],
            "linkedin": ["professional", "career", "business"],
            "instagram": ["inspiration", "lifestyle", "share"],
            "facebook": ["community", "discussion", "share"],
            "general": ["content", "social", "share"],
        }

        specific_tags = platform_hashtags.get(platform, platform_hashtags["general"])
        all_tags = base_tags + specific_tags

        hashtag_string = " ".join([f"#{tag}" for tag in all_tags])
        return hashtag_string

    def _generate_thread_point(self, context: WritingContext, index: int, total: int) -> str:
        """Generate a single thread point."""
        if index == 1:
            return f"Wprowadzenie: {context.topic} to temat, " "który mnie ostatnio fascynuje."
        elif index == total:
            return (
                f"🎯 Podsumowując: {context.topic} to temat, "
                "który zasługuje na uwagę. Co myślisz?"
            )
        else:
            return f"Punkt {index}: Kolejny aspekt {context.topic} " "wart przemyślenia."

    def _estimate_engagement(self, content: str) -> str:
        """Estimate engagement potential."""
        score = 0

        # Emojis increase engagement
        score += len(re.findall(r"[😀-🿿]", content)) * 2

        # Questions increase engagement
        score += content.count("?") * 3

        # Call-to-action words
        cta_words = ["komentarz", "udostępnij", "myślisz", "opinię"]
        score += sum(content.lower().count(word) for word in cta_words) * 2

        # Hashtags
        score += len(re.findall(r"#\w+", content))

        if score >= 15:
            return "High"
        elif score >= 8:
            return "Medium"
        else:
            return "Low"

    def score_content(self, content: str, context: WritingContext) -> dict[str, float]:
        """Score social media content."""
        scores = {}

        # Hook strength (presence of emojis, power words)
        hook_indicators = ["🔥", "💯", "⚡", "💡", "🎯"]
        scores["hook_strength"] = min(
            sum(content.count(indicator) for indicator in hook_indicators) * 20, 100
        )

        # Engagement potential
        engagement_score = self._estimate_engagement(content)
        scores["engagement"] = {"High": 90, "Medium": 70, "Low": 40}[engagement_score]

        # Platform optimization
        hashtag_count = len(re.findall(r"#\w+", content))
        scores["platform_optimization"] = min(hashtag_count * 15, 100)

        # Call to action presence
        cta_words = ["komentarz", "udostępnij", "myślisz", "opinię", "zapisz"]
        scores["call_to_action"] = min(
            sum(content.lower().count(word) for word in cta_words) * 25, 100
        )

        return scores
