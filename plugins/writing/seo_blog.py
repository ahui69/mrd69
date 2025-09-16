"""
SEO/Blog writing plugin.
"""

from __future__ import annotations

import re
from typing import Any

from .base import WritingContext, WritingPlugin, WritingResult


class SEOBlogPlugin(WritingPlugin):
    """Plugin for generating SEO-optimized blog content."""

    @property
    def name(self) -> str:
        return "seo_blog"

    @property
    def description(self) -> str:
        return "Generates SEO-optimized blog articles with proper structure"

    @property
    def category(self) -> str:
        return "seo"

    def validate_context(self, context: WritingContext) -> bool:
        """Check if context is suitable for SEO content."""
        topic = context.topic.lower()
        seo_indicators = [
            "blog",
            "artykuł",
            "seo",
            "website",
            "strona",
            "content",
            "jak",
            "co to",
            "dlaczego",
            "poradnik",
            "guide",
        ]
        return any(indicator in topic for indicator in seo_indicators)

    def get_default_style(self) -> dict[str, Any]:
        """SEO style: informative, structured, keyword-rich."""
        return {
            "pace": 0.4,  # Measured pace for readability
            "sensory": 0.5,  # Moderate sensory appeal
            "irony": 0.2,  # Professional tone
            "slang": 0.1,  # Minimal slang for credibility
            "pathos": 0.3,  # Some emotional connection
            "profanity": 0.0,  # Professional content
        }

    def generate(self, context: WritingContext) -> WritingResult:
        """Generate SEO-optimized blog content."""
        # Extract keywords and structure
        keywords = self._extract_keywords(context.topic)
        structure = self._plan_article_structure(context.topic)

        # Generate content sections
        content = self._generate_seo_article(context, keywords, structure)

        # Add SEO elements
        content = self._add_seo_elements(content, keywords)

        scoring = self.score_content(content, context)

        metadata = {
            "primary_keywords": keywords[:3],
            "word_count": len(content.split()),
            "readability_score": self._calculate_readability(content),
            "h1_count": content.count("# "),
            "h2_count": content.count("## "),
            "internal_links": content.count("["),
        }

        return WritingResult(content=content, metadata=metadata, scoring=scoring)

    def _extract_keywords(self, topic: str) -> list[str]:
        """Extract SEO keywords from topic."""
        # Clean and split topic
        words = re.findall(r"\b[a-ząćęłńóśźż]{3,}\b", topic.lower())

        # Remove common stop words
        stop_words = {
            "jak",
            "co",
            "to",
            "jest",
            "czy",
            "które",
            "gdzie",
            "kiedy",
            "dlaczego",
            "oraz",
            "ale",
            "lub",
            "więc",
            "też",
            "bardzo",
        }

        keywords = [word for word in words if word not in stop_words]

        # Add topic variations
        if "jak" in topic.lower():
            keywords.extend(["poradnik", "tutorial", "instrukcja"])

        if "co to" in topic.lower():
            keywords.extend(["definicja", "znaczenie", "wyjaśnienie"])

        return keywords[:10]

    def _plan_article_structure(self, topic: str) -> dict[str, list[str]]:
        """Plan article structure based on topic type."""
        topic_lower = topic.lower()

        if "jak" in topic_lower:
            return {
                "introduction": ["Czym jest", "Dlaczego warto", "Co zyskasz"],
                "main_sections": [
                    "Krok 1: Przygotowanie",
                    "Krok 2: Realizacja",
                    "Krok 3: Finalizacja",
                    "Najczęstsze błędy",
                    "Wskazówki ekspertów",
                ],
                "conclusion": ["Podsumowanie", "Następne kroki"],
            }
        elif "co to" in topic_lower:
            return {
                "introduction": ["Definicja", "Kontekst"],
                "main_sections": [
                    "Charakterystyka",
                    "Zastosowania",
                    "Zalety i wady",
                    "Przykłady praktyczne",
                ],
                "conclusion": ["Podsumowanie", "Dalsze informacje"],
            }
        else:
            return {
                "introduction": ["Wprowadzenie", "Cel artykułu"],
                "main_sections": [
                    "Główne aspekty",
                    "Analiza szczegółowa",
                    "Praktyczne zastosowanie",
                    "Przykłady",
                ],
                "conclusion": ["Wnioski", "Rekomendacje"],
            }

    def _generate_seo_article(
        self, context: WritingContext, keywords: list[str], structure: dict[str, list[str]]
    ) -> str:
        """Generate the main SEO article content."""
        primary_keyword = keywords[0] if keywords else context.topic.split()[0]

        # Article title (H1)
        title = self._generate_seo_title(context.topic, primary_keyword)
        content = [f"# {title}"]

        # Meta description hint
        content.append(
            f"<!-- Meta description: {self._generate_meta_description(context.topic, primary_keyword)} -->"
        )

        # Introduction
        content.append("## Wprowadzenie")
        intro = self._generate_introduction(context, primary_keyword)
        content.append(intro)

        # Main sections
        for section_title in structure["main_sections"]:
            content.append(f"## {section_title}")
            section_content = self._generate_section_content(section_title, context, keywords)
            content.append(section_content)

        # Conclusion
        content.append("## Podsumowanie")
        conclusion = self._generate_conclusion(context, primary_keyword)
        content.append(conclusion)

        # FAQ section for SEO
        content.append("## Najczęściej zadawane pytania")
        faq = self._generate_faq(context, keywords)
        content.append(faq)

        return "\n\n".join(content)

    def _generate_seo_title(self, topic: str, keyword: str) -> str:
        """Generate SEO-optimized title."""
        if "jak" in topic.lower():
            return f"Jak {keyword}: Kompletny poradnik krok po kroku"
        elif "co to" in topic.lower():
            return f"Co to jest {keyword}? Wszystko co musisz wiedzieć"
        else:
            return f"{keyword.title()}: Kompletny przewodnik 2024"

    def _generate_meta_description(self, topic: str, keyword: str) -> str:
        """Generate meta description."""
        return (
            f"Odkryj wszystko o {keyword}. Praktyczny przewodnik "
            f"z przykładami i wskazówkami ekspertów. Przeczytaj więcej!"
        )

    def _generate_introduction(self, context: WritingContext, keyword: str) -> str:
        """Generate article introduction."""
        intro_templates = {
            "energia": (
                f"🚀 {keyword.title()} to fascynujący temat, który może "
                f"całkowicie zmienić Twoje podejście! W tym artykule "
                f"odkryjesz wszystkie sekrety związane z {keyword}."
            ),
            "spokój": (
                f"{keyword.title()} to istotny element, który warto "
                f"dokładnie poznać. W niniejszym artykule przedstawię "
                f"kompleksowe informacje na temat {keyword}."
            ),
            "kreatywność": (
                f"Co jeśli powiem Ci, że {keyword} może być "
                f"kluczem do sukcesu? Ten artykuł to nietypowe "
                f"spojrzenie na {keyword} z perspektywy praktyka."
            ),
        }

        mood = context.mood if context.mood in intro_templates else "spokój"
        base_intro = intro_templates[mood]

        # Add value proposition
        value_prop = (
            f"\n\nPo przeczytaniu tego artykułu będziesz wiedział:\n"
            f"- Wszystkie kluczowe aspekty {keyword}\n"
            f"- Praktyczne zastosowania\n"
            f"- Najlepsze praktyki i wskazówki ekspertów"
        )

        return base_intro + value_prop

    def _generate_section_content(
        self, section_title: str, context: WritingContext, keywords: list[str]
    ) -> str:
        """Generate content for a specific section."""
        primary_keyword = keywords[0] if keywords else "temat"

        # Basic content generation based on section type
        if "krok" in section_title.lower():
            return (
                f"W tym etapie pracy z {primary_keyword} kluczowe jest "
                f"systematyczne podejście. Oto co powinieneś zrobić:\n\n"
                f"1. **Analiza sytuacji** - dokładnie przeanalizuj "
                f"wszystkie aspekty związane z {primary_keyword}\n"
                f"2. **Planowanie działań** - stwórz konkretny plan "
                f"implementacji\n"
                f"3. **Realizacja** - wprowadź zaplanowane zmiany "
                f"w życie\n\n"
                f"💡 **Wskazówka:** Pamiętaj, że {primary_keyword} "
                f"wymaga cierpliwości i konsekwentnego działania."
            )

        elif "błędy" in section_title.lower():
            return (
                f"Pracując z {primary_keyword}, wiele osób popełnia "
                f"te same błędy. Oto najczęstsze z nich:\n\n"
                f"❌ **Błąd #1:** Zbyt szybkie tempo - {primary_keyword} "
                f"wymaga czasu\n"
                f"❌ **Błąd #2:** Brak systematyczności w podejściu\n"
                f"❌ **Błąd #3:** Ignorowanie podstawowych zasad\n\n"
                f"Unikając tych błędów, znacznie zwiększysz swoje "
                f"szanse na sukces z {primary_keyword}."
            )

        else:
            return (
                f"Analizując {primary_keyword}, warto zwrócić uwagę "
                f"na kilka kluczowych aspektów. To właśnie one decydują "
                f"o skuteczności całego procesu.\n\n"
                f"Badania pokazują, że osoby, które stosują właściwe "
                f"podejście do {primary_keyword}, osiągają lepsze "
                f"rezultaty w krótszym czasie."
            )

    def _generate_conclusion(self, context: WritingContext, keyword: str) -> str:
        """Generate article conclusion."""
        return (
            f"{keyword.title()} to bez wątpienia temat warty uwagi. "
            f"Jak pokazałem w tym artykule, właściwe podejście do "
            f"{keyword} może przynieść znaczące korzyści.\n\n"
            f"**Kluczowe wnioski:**\n"
            f"- {keyword.title()} wymaga systematycznego podejścia\n"
            f"- Unikanie typowych błędów jest kluczowe\n"
            f"- Praktyczne zastosowanie teorii daje najlepsze rezultaty\n\n"
            f"Czy masz doświadczenie z {keyword}? Podziel się swoimi "
            f"spostrzeżeniami w komentarzach!"
        )

    def _generate_faq(self, context: WritingContext, keywords: list[str]) -> str:
        """Generate FAQ section for SEO."""
        primary_keyword = keywords[0] if keywords else "temat"

        faq_items = [
            (
                f"Jak zacząć pracę z {primary_keyword}?",
                f"Najlepiej zacząć od dokładnego zapoznania się z podstawami "
                f"{primary_keyword}. Polecam rozpoczęcie od prostych ćwiczeń.",
            ),
            (
                f"Ile czasu zajmuje opanowanie {primary_keyword}?",
                f"Czas potrzebny na opanowanie {primary_keyword} zależy od "
                f"indywidualnych predyspozycji, ale zazwyczaj to kilka tygodni "
                f"regularnej praktyki.",
            ),
            (
                f"Jakie są najczęstsze błędy związane z {primary_keyword}?",
                "Najczęstsze błędy to zbyt szybkie tempo nauki, brak "
                "systematyczności oraz ignorowanie podstawowych zasad.",
            ),
        ]

        faq_content = []
        for question, answer in faq_items:
            faq_content.append(f"### {question}")
            faq_content.append(answer)

        return "\n\n".join(faq_content)

    def _add_seo_elements(self, content: str, keywords: list[str]) -> str:
        """Add additional SEO elements."""
        # Add internal linking suggestions
        seo_section = "\n\n---\n\n### SEO & Linki wewnętrzne\n\n"
        seo_section += "**Sugerowane linki wewnętrzne:**\n"

        for keyword in keywords[:3]:
            seo_section += f"- [Więcej o {keyword}](/{keyword.replace(' ', '-')})\n"

        # Add schema markup suggestion
        seo_section += "\n**Schema markup:** Article, FAQ\n"

        # Add related keywords
        seo_section += f"\n**Powiązane słowa kluczowe:** {', '.join(keywords[:8])}\n"

        return content + seo_section

    def _calculate_readability(self, content: str) -> float:
        """Calculate basic readability score."""
        # Simple readability calculation
        sentences = len(re.findall(r"[.!?]+", content))
        words = len(content.split())

        if sentences == 0:
            return 50.0

        avg_words_per_sentence = words / sentences

        # Flesch-like scoring
        if avg_words_per_sentence <= 15:
            return 90.0
        elif avg_words_per_sentence <= 20:
            return 75.0
        elif avg_words_per_sentence <= 25:
            return 60.0
        else:
            return 40.0

    def score_content(self, content: str, context: WritingContext) -> dict[str, float]:
        """Score SEO blog content."""
        scores = {
            "seo_optimization": 0.0,
            "readability": 0.0,
            "structure": 0.0,
            "keyword_density": 0.0,
            "engagement": 0.0,
        }

        # SEO optimization
        h1_count = content.count("# ")
        h2_count = content.count("## ")

        if h1_count == 1 and h2_count >= 3:
            scores["seo_optimization"] = 90.0
        elif h1_count == 1 and h2_count >= 1:
            scores["seo_optimization"] = 70.0
        else:
            scores["seo_optimization"] = 40.0

        # Readability
        scores["readability"] = self._calculate_readability(content)

        # Structure
        required_elements = ["wprowadzenie", "podsumowanie", "##"]
        structure_score = sum(30 for elem in required_elements if elem.lower() in content.lower())
        scores["structure"] = min(100.0, structure_score)

        # Keyword density (basic check)
        word_count = len(content.split())
        if word_count > 500:
            scores["keyword_density"] = 80.0
        elif word_count > 300:
            scores["keyword_density"] = 65.0
        else:
            scores["keyword_density"] = 45.0

        # Engagement elements
        engagement_markers = ["💡", "❌", "✅", "🚀", "pytania", "komentarzach"]
        engagement = sum(15 for marker in engagement_markers if marker in content)
        scores["engagement"] = min(100.0, engagement)

        return scores
