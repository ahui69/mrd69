"""
Vinted marketplace writing plugin.
"""

from __future__ import annotations

import re
from typing import Any

from .base import WritingContext, WritingPlugin, WritingResult


class VintedPlugin(WritingPlugin):
    """Plugin for generating Vinted auction descriptions."""

    @property
    def name(self) -> str:
        return "vinted"

    @property
    def description(self) -> str:
        return "Generates professional Vinted auction descriptions with SEO and price suggestions"

    @property
    def category(self) -> str:
        return "marketplace"

    def validate_context(self, context: WritingContext) -> bool:
        """Check if context contains marketplace-relevant info."""
        topic = context.topic.lower()
        marketplace_indicators = [
            "vinted",
            "sprzedam",
            "aukcja",
            "rozmiar",
            "stan",
            "marka",
            "brand",
            "cena",
            "oryginał",
            "vintage",
            "secondhand",
            "ubranie",
            "buty",
        ]
        return any(indicator in topic for indicator in marketplace_indicators)

    def get_default_style(self) -> dict[str, Any]:
        """Marketplace style: trustworthy, detailed, SEO-friendly."""
        return {
            "pace": 0.4,  # Measured pace for credibility
            "sensory": 0.6,  # Some sensory details for appeal
            "irony": 0.1,  # Minimal irony for trust
            "slang": 0.2,  # Light casual language
            "pathos": 0.5,  # Moderate emotional appeal
            "profanity": 0.0,  # Professional marketplace
        }

    def generate(self, context: WritingContext) -> WritingResult:
        """Generate Vinted auction description."""
        # Extract item details from topic
        details = self._extract_item_details(context.topic)

        # Generate core description
        content = self._generate_description(details, context)

        # Add SEO elements
        content = self._add_seo_elements(content, details)

        # Calculate scoring
        scoring = self.score_content(content, context)

        metadata = {
            "extracted_details": details,
            "category": "marketplace",
            "platform": "vinted",
            "price_estimated": details.get("price_estimate"),
        }

        return WritingResult(content=content, metadata=metadata, scoring=scoring)

    def _extract_item_details(self, topic: str) -> dict[str, Any]:
        """Extract item details from topic description."""
        details = {
            "brand": "",
            "item_type": "",
            "size": "",
            "condition": "",
            "color": "",
            "material": "",
            "model": "",
            "price_estimate": None,
        }

        # Brand extraction
        brands = [
            "nike",
            "adidas",
            "puma",
            "reebok",
            "converse",
            "vans",
            "h&m",
            "zara",
            "reserved",
            "cropp",
            "house",
            "mohito",
            "levi's",
            "tommy hilfiger",
            "calvin klein",
            "hugo boss",
            "gucci",
            "prada",
            "versace",
            "armani",
            "dolce",
            "gabbana",
        ]

        topic_lower = topic.lower()
        for brand in brands:
            if brand in topic_lower:
                details["brand"] = brand.title()
                break

        # Size extraction
        size_patterns = [
            r"\b(xs|s|m|l|xl|xxl|xxxl)\b",
            r"\b(\d{2,3})\b",  # Numeric sizes
            r"\b(3[0-9]|4[0-9]|5[0-1])\b",  # Shoe sizes
        ]

        for pattern in size_patterns:
            match = re.search(pattern, topic_lower)
            if match:
                details["size"] = match.group(1).upper()
                break

        # Condition extraction
        conditions = ["nowy", "bardzo dobry", "dobry", "zadowalający"]
        for condition in conditions:
            if condition in topic_lower:
                details["condition"] = condition
                break

        # Color extraction
        colors = [
            "czarny",
            "biały",
            "szary",
            "czerwony",
            "niebieski",
            "zielony",
            "żółty",
            "różowy",
            "fioletowy",
            "brązowy",
            "beżowy",
            "granatowy",
        ]
        for color in colors:
            if color in topic_lower:
                details["color"] = color
                break

        return details

    def _generate_description(self, details: dict[str, Any], context: WritingContext) -> str:
        """Generate the main auction description."""
        brand = details.get("brand", "")
        item_type = self._infer_item_type(context.topic)
        size = details.get("size", "")
        condition = details.get("condition", "bardzo dobry")
        color = details.get("color", "")

        # Title
        title_parts = [p for p in [brand, item_type, size] if p]
        title = " ".join(title_parts)

        description = f"# {title}\n\n"

        # Main details
        description += f"**Marka:** {brand or 'Nie określono'}\n"
        description += f"**Typ:** {item_type}\n"
        description += f"**Rozmiar:** {size or 'Zobacz wymiary'}\n"
        description += f"**Stan:** {condition}\n"

        if color:
            description += f"**Kolor:** {color}\n"

        description += "\n"

        # Description body based on mood/style
        if context.mood == "energia":
            description += self._generate_energetic_description(brand, item_type)
        elif context.mood == "spokój":
            description += self._generate_calm_description(brand, item_type)
        else:
            description += self._generate_balanced_description(brand, item_type)

        # Care instructions
        description += "\n\n## Pielęgnacja\n"
        description += "Prać zgodnie z metką. Szczegóły pielęgnacji na żądanie.\n"

        # Shipping info
        description += "\n## Wysyłka i zwroty\n"
        description += "📦 Wysyłka w 24-48h\n"
        description += "📋 Opis zgodny ze stanem rzeczywistym\n"
        description += "🏠 Środowisko wolne od dymu\n"
        description += "💰 Bundle discount -15% przy zakupie 2+ rzeczy\n"

        return description

    def _infer_item_type(self, topic: str) -> str:
        """Infer item type from topic."""
        topic_lower = topic.lower()

        clothing_items = {
            "koszulka": ["koszulka", "t-shirt", "tshirt"],
            "spodnie": ["spodnie", "jeansy", "dżinsy"],
            "sukienka": ["sukienka", "dress"],
            "bluza": ["bluza", "hoodie", "sweatshirt"],
            "kurtka": ["kurtka", "jacket", "płaszcz"],
            "buty": ["buty", "shoes", "sneakers", "adidasy"],
            "sweter": ["sweter", "sweater", "kardigan"],
            "szorty": ["szorty", "shorts"],
            "spódnica": ["spódnica", "skirt"],
        }

        for item_type, keywords in clothing_items.items():
            if any(keyword in topic_lower for keyword in keywords):
                return item_type

        return "ubranie"

    def _generate_energetic_description(self, brand: str, item_type: str) -> str:
        """Generate energetic style description."""
        descriptions = [
            f"🔥 **Absolutny must-have!** {brand} {item_type} który nie może Cię zawieść!",
            "✨ To NIE jest kolejna zwykła rzecz - to **statement piece** który mówi za Ciebie!",
            "💫 Jakość **premium** w cenie, która nie rujnuje budżetu. Sprawdź sam!",
        ]

        if brand.lower() in ["nike", "adidas"]:
            descriptions.append("🏃‍♀️ Sport meets style - idealne na trening i na miasto!")

        return "\n".join(descriptions[:2])

    def _generate_calm_description(self, brand: str, item_type: str) -> str:
        """Generate calm, professional description."""
        return (
            f"Oferuję {item_type} marki {brand} w bardzo dobrym stanie. "
            f"Rzecz była noszona sporadycznie i starannie przechowywana. "
            f"Idealna dla osób ceniących jakość i klasyczny styl."
        )

    def _generate_balanced_description(self, brand: str, item_type: str) -> str:
        """Generate balanced description."""
        return (
            f"Super {item_type} od {brand}! Świetna jakość i wygoda. "
            f"Stan jak na zdjęciach - wszystkie detale widoczne. "
            f"Idealny do codziennych stylizacji. Polecam! 😊"
        )

    def _add_seo_elements(self, content: str, details: dict[str, Any]) -> str:
        """Add SEO hashtags and keywords."""
        seo_section = "\n\n---\n\n### SEO & Hashtagi\n\n"

        keywords = []

        # Brand keywords
        if details.get("brand"):
            keywords.extend([details["brand"], details["brand"].lower()])

        # Generic keywords
        keywords.extend(
            [
                "oryginał",
                "100%autentyk",
                "vintage",
                "secondhand",
                "polecam",
                "okazja",
                "tanio",
                "szybka wysyłka",
            ]
        )

        # Size keywords
        if details.get("size"):
            keywords.append(f"rozmiar{details['size']}")

        # Color keywords
        if details.get("color"):
            keywords.append(details["color"])

        # Remove duplicates and create hashtags
        unique_keywords = list(dict.fromkeys(keywords))
        hashtags = [f"#{kw.replace(' ', '').lower()}" for kw in unique_keywords[:15]]

        seo_section += " ".join(hashtags)

        return content + seo_section

    def score_content(self, content: str, context: WritingContext) -> dict[str, float]:
        """Score Vinted auction content."""
        scores = {
            "coherence": 80.0,
            "seo_optimization": 0.0,
            "trustworthiness": 0.0,
            "completeness": 0.0,
            "engagement": 0.0,
        }

        # SEO scoring
        hashtag_count = content.count("#")
        if hashtag_count >= 10:
            scores["seo_optimization"] = 90.0
        elif hashtag_count >= 5:
            scores["seo_optimization"] = 70.0
        else:
            scores["seo_optimization"] = 40.0

        # Trustworthiness scoring
        trust_indicators = ["stan", "wysyłka", "oryginał", "autentyk", "zgodny"]
        trust_score = sum(20 for indicator in trust_indicators if indicator in content.lower())
        scores["trustworthiness"] = min(100.0, trust_score)

        # Completeness scoring
        required_sections = ["rozmiar", "stan", "wysyłka", "pielęgnacja"]
        completeness = sum(25 for section in required_sections if section in content.lower())
        scores["completeness"] = min(100.0, completeness)

        # Engagement scoring
        engagement_markers = ["!", "🔥", "✨", "💫", "😊", "must-have", "polecam"]
        engagement = sum(10 for marker in engagement_markers if marker in content)
        scores["engagement"] = min(100.0, engagement)

        return scores
