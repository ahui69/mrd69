"""
Plugin manager for dynamically loading and managing writing plugins.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

from .base import WritingContext, WritingPlugin, WritingResult


class PluginManager:
    """Manages loading and execution of writing plugins."""

    def __init__(self, plugins_dir: str | Path | None = None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else Path(__file__).parent
        self.plugins: dict[str, WritingPlugin] = {}
        self.categories: dict[str, list[str]] = {}
        self._load_plugins()

    def _load_plugins(self) -> None:
        """Load all plugins from the plugins directory."""
        if not self.plugins_dir.exists():
            return

        for file in self.plugins_dir.glob("*.py"):
            if file.name.startswith("_") or file.name in ("base.py", "__init__.py", "manager.py"):
                continue

            try:
                # Import module
                module_name = f"plugins.writing.{file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find plugin classes
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, WritingPlugin)
                            and obj != WritingPlugin
                        ):

                            plugin = obj()
                            self.plugins[plugin.name] = plugin

                            # Organize by category
                            category = plugin.category
                            if category not in self.categories:
                                self.categories[category] = []
                            self.categories[category].append(plugin.name)

            except Exception as e:
                print(f"Failed to load plugin {file}: {e}")

    def get_plugin(self, name: str) -> WritingPlugin | None:
        """Get plugin by name."""
        return self.plugins.get(name)

    def list_plugins(self, category: str | None = None) -> list[str]:
        """List available plugins, optionally filtered by category."""
        if category:
            return self.categories.get(category, [])
        return list(self.plugins.keys())

    def generate_content(self, plugin_name: str, context: WritingContext) -> WritingResult | None:
        """Generate content using specified plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return None

        if not plugin.validate_context(context):
            return None

        return plugin.generate(context)

    def generate_variants(
        self,
        plugin_name: str,
        context: WritingContext,
        variant_styles: list[dict[str, Any]] | None = None,
    ) -> list[WritingResult]:
        """Generate multiple variants with different styles."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return []

        # Default variant styles: professional, hype, relaxed
        if not variant_styles:
            variant_styles = [
                {"pace": 0.3, "irony": 0.1, "slang": 0.0, "pathos": 0.2},  # professional
                {"pace": 0.8, "irony": 0.4, "slang": 0.6, "pathos": 0.8},  # hype
                {"pace": 0.6, "irony": 0.5, "slang": 0.4, "pathos": 0.3},  # relaxed
            ]

        results = []
        for style in variant_styles:
            variant_context = WritingContext(
                topic=context.topic,
                user=context.user,
                persona=context.persona,
                tone=context.tone,
                audience=context.audience,
                keywords=context.keywords,
                style_params={**context.style_params, **style},
                research_enabled=context.research_enabled,
                mood=context.mood,
                energy=context.energy,
                creativity=context.creativity,
            )

            result = plugin.generate(variant_context)
            if result:
                results.append(result)

        return results

    def auto_select_plugin(self, context: WritingContext) -> str | None:
        """Auto-select best plugin based on context."""
        # Simple heuristics for plugin selection
        topic_lower = context.topic.lower()

        # Marketplace detection
        marketplace_keywords = [
            "vinted",
            "olx",
            "allegro",
            "ebay",
            "sprzedam",
            "aukcja",
            "brand",
            "rozmiar",
            "stan",
            "cena",
        ]
        if any(kw in topic_lower for kw in marketplace_keywords):
            marketplace_plugins = self.categories.get("marketplace", [])
            if marketplace_plugins:
                return marketplace_plugins[0]

        # Social media detection
        social_keywords = [
            "post",
            "twitter",
            "instagram",
            "facebook",
            "linkedin",
            "thread",
            "wątek",
            "social",
        ]
        if any(kw in topic_lower for kw in social_keywords):
            social_plugins = self.categories.get("social", [])
            if social_plugins:
                return social_plugins[0]

        # SEO/Blog detection
        seo_keywords = ["blog", "seo", "artykuł", "website", "strona", "content"]
        if any(kw in topic_lower for kw in seo_keywords):
            seo_plugins = self.categories.get("seo", [])
            if seo_plugins:
                return seo_plugins[0]

        # Email detection
        email_keywords = ["email", "mail", "newsletter", "kampania"]
        if any(kw in topic_lower for kw in email_keywords):
            email_plugins = self.categories.get("email", [])
            if email_plugins:
                return email_plugins[0]

        # Default to creative
        creative_plugins = self.categories.get("creative", [])
        if creative_plugins:
            return creative_plugins[0]

        # Fallback to first available plugin
        if self.plugins:
            return list(self.plugins.keys())[0]

        return None


# Global plugin manager instance
_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
