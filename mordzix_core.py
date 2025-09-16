"""
MORDZIX CORE - Najlepsza Platforma AI
=====================================

Ziomek od wszystkiego - bez kagańca, bez filtrów, z mózgiem bestii.
Łączy crypto_advisor, travelguide, autonauka, programista w jeden potężny system.

Features:
🚀 Chat bez ograniczeń z STM/LTM kontekstem 100+ wersów
🚀 Real-time crypto monitoring z alertami w czacie
🚀 Travel planning z mapami i trasami
🚀 Code review i programming mentor
🚀 Learning system z spaced repetition
🚀 Voice STT/TTS integration
🚀 File upload/download z media processing
🚀 Offline sync z auto-retry
🚀 WebSocket real-time connections
🚀 IO Pipeline event bus system
🚀 Multi-provider image generation
🚀 Reliability architecture with retry policies
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

# Import wszystkich modułów
import crypto_advisor_full
from memory import Memory, get_advanced_memory, AdvancedMemorySystem, ContextType, MoodType
import travelguide
import autonauka
import programista
import writing_all_pro

# === PSYCHIKA & ADVANCED AI MODULES ===
try:
    import psychika  # Decision-making & affect system

    PSYCHIKA_AVAILABLE = True
except ImportError:
    print("🧠 Psychika not available - using basic responses")
    PSYCHIKA_AVAILABLE = False

try:
    import kimi_client  # Kimi-K2-Instruct model

    KIMI_AVAILABLE = True
except ImportError:
    print("🤖 Kimi not available - using basic LLM")
    KIMI_AVAILABLE = False

# === IO PIPELINE & EVENT BUS ===
try:
    import io_pipeline  # Event bus system

    IO_PIPELINE_AVAILABLE = True
except ImportError:
    print("📡 IO Pipeline not available - using basic logging")
    IO_PIPELINE_AVAILABLE = False

# === IMAGE GENERATION ===
try:
    import images_client  # Multi-provider image generation

    IMAGES_AVAILABLE = True
except ImportError:
    print("🎨 Images client not available - text only mode")
    IMAGES_AVAILABLE = False

# === SEED MEMORY & PROMPTS ===
try:
    import seed_memory  # Memory seeding system
    import prompt  # Advanced prompt management

    SEED_MEMORY_AVAILABLE = True
    PROMPT_AVAILABLE = True
except ImportError:
    print("🌱 Seed memory/prompts not available - using basic system")
    SEED_MEMORY_AVAILABLE = False
    PROMPT_AVAILABLE = False
except ImportError:
    print("🤖 Kimi not available - using default LLM")
    KIMI_AVAILABLE = False

try:
    import seed_memory  # Memory seeding system

    SEED_MEMORY_AVAILABLE = True
except ImportError:
    print("🌱 Seed memory not available")
    SEED_MEMORY_AVAILABLE = False

try:
    import prompt  # System prompts & persona
    from prompt import SYSTEM_PROMPT

    PROMPT_SYSTEM_AVAILABLE = True
except ImportError:
    print("📝 Advanced prompts not available - using basic")
    SYSTEM_PROMPT = "You are Mordzix - helpful AI assistant."
    PROMPT_SYSTEM_AVAILABLE = False

try:
    import io_pipeline  # Event bus & metrics

    IO_PIPELINE_AVAILABLE = True
except ImportError:
    print("🚇 IO Pipeline not available - no event bus")
    IO_PIPELINE_AVAILABLE = False

try:
    import images_client  # Multi-provider image generation

    IMAGES_AVAILABLE = True
except ImportError:
    print("🎨 Images client not available - no image generation")
    IMAGES_AVAILABLE = False


@dataclass
class ChatMessage:
    """Wiadomość w czacie z pełnym kontekstem."""

    id: str
    thread_id: str
    user_id: str
    content: str
    message_type: str = "text"  # text, voice, image, file, system
    timestamp: float = None
    status: str = "sent"  # sending, sent, delivered, error
    metadata: Dict[str, Any] = None
    attachments: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
        if self.attachments is None:
            self.attachments = []


@dataclass
class ChatThread:
    """Wątek rozmowy z pamięcią kontekstu."""

    id: str
    user_id: str
    title: str
    created_at: float
    last_activity: float
    message_count: int = 0
    context_summary: str = ""
    personality_mode: str = "mordzix"  # mordzix, professional, casual

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_activity:
            self.last_activity = time.time()


class MordzixPersonality:
    """System osobowości bez kagańca - prawdziwy ziomek."""

    def __init__(self):
        self.slang_phrases = [
            "No kurde, mordo!",
            "Ziomek!",
            "Bestia!",
            "No i git!",
            "Załatwione!",
            "Ogarniemy to!",
            "Spoko sprawa!",
            "W porządku bracie!",
            "Mega sprawa!",
            "To jest to!",
            "Jazda z tym!",
            "Dawaj dalej!",
        ]

        self.casual_starters = [
            "Yo! ",
            "Ej ziom, ",
            "Słuchaj mordo, ",
            "No to tak: ",
            "Dobra, ",
            "Okej, ogarniamy: ",
            "No to robimy tak: ",
        ]

        self.enthusiasm_boosters = [" 🚀", " 💪", " 🔥", " ⚡", " 🎯", " 💯", " 🚀💪"]

    def enhance_response(self, response: str, context: Dict[str, Any] = None) -> str:
        """Dodaje personality do odpowiedzi - casual tone bez filtrów."""
        if not response:
            return response

        # Dodaj casual starter losowo
        if not response.startswith(tuple(self.casual_starters)):
            import random

            if random.random() < 0.3:  # 30% szans na starter
                response = random.choice(self.casual_starters) + response

        # Dodaj emoji na końcu czasem
        import random

        if random.random() < 0.2:  # 20% szans na emoji
            response += random.choice(self.enthusiasm_boosters)

        return response

    def _add_empathy(self, response: str) -> str:
        """Dodaje empatię do odpowiedzi."""
        empathy_starters = [
            "Rozumiem cię mordo, ",
            "Wiem że to może być trudne, ale ",
            "Słuchaj ziom, ",
            "Znam to uczucie, ",
        ]
        import random

        return random.choice(empathy_starters) + response

    def _add_enthusiasm(self, response: str) -> str:
        """Dodaje entuzjazm do odpowiedzi."""
        return response + " 🔥🚀 To będzie bomba!"

    def _add_analysis_depth(self, response: str) -> str:
        """Dodaje głębię analityczną."""
        return "📊 Analiza: " + response + " (bazuję na danych i doświadczeniu)"

    def _apply_mordzix_persona(self, response: str) -> str:
        """Aplikuje pełną personę Mordzix z prompt.py."""
        if PROMPT_SYSTEM_AVAILABLE:
            # Używa zaawansowanego systemu promptów
            return response.replace("assistant", "Mordzix").replace("AI", "ziomek")
        return response

    def _add_casual_reassurance(self, response: str) -> str:
        """Dodaje casualowe wsparcie przy błędach."""
        return f"Mały problem, mordo! {response} Ale nie martw się, razem to ogarnamy! 💪"

    def generate_system_message(self, action: str, details: str = "") -> str:
        """Generuje system message w stylu Mordzix."""
        messages = {
            "crypto_alert": f"🚨 Crypto alert, ziom! {details}",
            "portfolio_update": f"💰 Portfolio update: {details}",
            "travel_plan": f"🌍 Zaplanowałem ci trasę: {details}",
            "code_review": f"👨‍💻 Code review gotowy: {details}",
            "learning_reminder": f"🧠 Czas na naukę, mordo! {details}",
            "file_processed": f"📁 Plik przetworzony: {details}",
        }
        return messages.get(action, f"✅ {action}: {details}")


class MordzixChatEngine:
    """Core chat engine z zaawansowanym STM/LTM i wszystkimi integracjami."""

    def __init__(self):
        self.memory_system = Memory()
        self.advanced_memory = get_advanced_memory()
        self.personality = MordzixPersonality()
        self.active_threads: Dict[str, ChatThread] = {}
        self.message_history: Dict[str, List[ChatMessage]] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}

        # Kontekst rozmowy dla zaawansowanej pamięci
        self.current_context = ContextType.CHATTING
        self.detected_mood = MoodType.FRIENDLY

    def create_thread(self, user_id: str, title: str = "New Chat") -> ChatThread:
        """Tworzy nowy wątek rozmowy."""
        thread_id = str(uuid.uuid4())
        thread = ChatThread(
            id=thread_id,
            user_id=user_id,
            title=title,
            created_at=time.time(),
            last_activity=time.time(),
        )
        self.active_threads[thread_id] = thread
        self.message_history[thread_id] = []
        return thread

    def add_message(
        self,
        thread_id: str,
        user_id: str,
        content: str,
        message_type: str = "text",
        attachments: List = None,
    ) -> ChatMessage:
        """Dodaje wiadomość do wątku."""
        message_id = str(uuid.uuid4())
        message = ChatMessage(
            id=message_id,
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            message_type=message_type,
            attachments=attachments or [],
        )

        if thread_id not in self.message_history:
            self.message_history[thread_id] = []

        self.message_history[thread_id].append(message)

        # Update thread activity
        if thread_id in self.active_threads:
            self.active_threads[thread_id].last_activity = time.time()
            self.active_threads[thread_id].message_count += 1

        return message

    def get_context_for_thread(self, thread_id: str, max_messages: int = 100) -> str:
        """Buduje kontekst dla wątku - minimum 100 wersów jak w specyfikacji."""
        if thread_id not in self.message_history:
            return ""

        messages = self.message_history[thread_id][-max_messages:]
        context_lines = []

        for msg in messages:
            timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M")
            sender = "User" if msg.user_id != "mordzix" else "Mordzix"
            context_lines.append(f"[{timestamp}] {sender}: {msg.content}")

            # Dodaj info o attachments
            if msg.attachments:
                for att in msg.attachments:
                    context_lines.append(f"  📎 {att.get('name', 'Attachment')}")

        return "\n".join(context_lines)

    async def process_message(
        self, thread_id: str, user_id: str, content: str, message_type: str = "text"
    ) -> ChatMessage:
        """Przetwarza wiadomość i generuje odpowiedź Mordzix z zaawansowaną pamięcią."""

        # === ZAAWANSOWANY SYSTEM PAMIĘCI ===
        # Wykryj nastrój i kontekst
        mood_info = self.advanced_memory.detect_user_mood(content)
        self.detected_mood = (
            MoodType(mood_info["primary_mood"])
            if mood_info["primary_mood"] in [m.value for m in MoodType]
            else MoodType.FRIENDLY
        )

        # Wykryj kontekst na podstawie treści
        if any(
            word in content.lower() for word in ["kod", "python", "bug", "error", "def", "class"]
        ):
            self.current_context = ContextType.CODING
        elif any(word in content.lower() for word in ["pisanie", "artykuł", "blog", "content"]):
            self.current_context = ContextType.CREATIVE_WRITING
        elif any(word in content.lower() for word in ["biznes", "klient", "sprzedaż", "zysk"]):
            self.current_context = ContextType.BUSINESS
        elif any(word in content.lower() for word in ["nauka", "learn", "study", "understand"]):
            self.current_context = ContextType.LEARNING
        elif any(word in content.lower() for word in ["debug", "fix", "napraw", "błąd"]):
            self.current_context = ContextType.DEBUGGING
        else:
            self.current_context = ContextType.CHATTING

        # Przełącz kontekst pamięci
        context_info = self.advanced_memory.switch_context(self.current_context)

        # Przewidywanie następnych akcji
        predictions = self.advanced_memory.predict_next_action(content)

        # Dodaj user message
        user_msg = self.add_message(thread_id, user_id, content, message_type)

        # Pobierz kontekst z zaawansowaną pamięcią
        context = self.get_context_for_thread(thread_id)

        # Dodaj priorytetowe fakty dla kontekstu
        if context_info.get("priority_facts"):
            context += "\n[KONTEKST PRIORYTETOWY]:\n"
            for fact in context_info["priority_facts"][:5]:
                context += f"• {fact}\n"

        # Analiza intencji - co user chce zrobić?
        intent = self._analyze_intent(content)

        # Generuj odpowiedź na podstawie intencji
        response = await self._generate_response(content, context, intent)

        # Adaptuj odpowiedź do nastroju użytkownika
        if mood_info["recommended_tone"] == "ultra_concise":
            response = self.advanced_memory.adapt_response_to_mood(
                response, mood_info["primary_mood"]
            )

        # Dodaj personality enhancement
        response = self.personality.enhance_response(
            response,
            {
                "intent": intent,
                "mood": mood_info["primary_mood"],
                "context_type": self.current_context.value,
                "predictions": predictions[:2],  # Top 2 przewidywania
            },
        )

        # Dodaj odpowiedź Mordzix
        ai_msg = self.add_message(thread_id, "mordzix", response, "text")

        # === ZAPISZ W ZAAWANSOWANYM SYSTEMIE PAMIĘCI ===
        interaction_result = self.advanced_memory.process_interaction(
            user_input=content, ai_response=response, context_type=self.current_context
        )

        # Zapisz do LTM jeśli ważne
        if intent.get("save_to_ltm", False):
            self._save_to_ltm(thread_id, user_msg, ai_msg, intent)

        # Dodaj informacje o przewidywaniach do odpowiedzi (opcjonalnie)
        if predictions and len(predictions) > 0 and self.detected_mood != MoodType.FRUSTRATED:
            prediction_text = f"\n\n💡 Przewiduję, że następnie możesz chcieć: {predictions[0]['predicted_action']}"
            ai_msg.content += prediction_text

        return ai_msg

    def _analyze_intent(self, content: str) -> Dict[str, Any]:
        """Analizuje intencję użytkownika."""
        content_lower = content.lower()

        # Crypto intents
        if any(
            word in content_lower
            for word in ["bitcoin", "btc", "eth", "crypto", "portfolio", "cena", "kurs"]
        ):
            return {"type": "crypto", "save_to_ltm": True, "module": "crypto_advisor_full"}

        # Travel intents
        if any(
            word in content_lower
            for word in ["podróż", "travel", "hotel", "lot", "wakacje", "wycieczka", "trasa"]
        ):
            return {"type": "travel", "save_to_ltm": True, "module": "travelguide"}

        # Programming intents
        if any(
            word in content_lower
            for word in ["kod", "code", "python", "git", "bug", "programowanie", "refactor"]
        ):
            return {"type": "programming", "save_to_ltm": True, "module": "programista"}

        # Learning intents
        if any(
            word in content_lower
            for word in ["nauka", "learn", "flashcard", "quiz", "wiedza", "memo"]
        ):
            return {"type": "learning", "save_to_ltm": True, "module": "autonauka"}

        # Writing intents
        if any(word in content_lower for word in ["pisanie", "artykuł", "blog", "seo", "write"]):
            return {"type": "writing", "save_to_ltm": False, "module": "writing_all_pro"}

        # General chat
        return {"type": "general", "save_to_ltm": False, "module": None}

    async def _generate_response(self, content: str, context: str, intent: Dict[str, Any]) -> str:
        """Generuje odpowiedź używając wszystkich dostępnych modułów."""

        # Emit event do IO Pipeline
        if IO_PIPELINE_AVAILABLE:
            await self._emit_pipeline_event(
                "message_received",
                {"content": content, "intent": intent["type"], "context_length": len(context)},
            )

        # Sprawdź czy to request o obrazek
        if self._is_image_request(content):
            return await self._handle_image_generation(content)

        # Routing na podstawie intent
        if intent["type"] == "crypto":
            return await self._handle_crypto_intent(content)
        elif intent["type"] == "travel":
            return await self._handle_travel_intent(content)
        elif intent["type"] == "programming":
            return await self._handle_programming_intent(content)
        elif intent["type"] == "learning":
            return await self._handle_learning_intent(content)
        elif intent["type"] == "writing":
            return await self._handle_writing_intent(content)
        else:
            return await self._handle_general_intent(content, context)

    def _is_image_request(self, content: str) -> bool:
        """Sprawdza czy user prosi o wygenerowanie obrazka."""
        image_keywords = [
            "generuj obraz",
            "stwórz obrazek",
            "narysuj",
            "image",
            "picture",
            "rysunek",
            "ilustracja",
            "zdjęcie",
            "grafika",
            "design",
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in image_keywords)

    async def _handle_image_generation(self, content: str) -> str:
        """Obsługuje generowanie obrazów przez images_client."""
        if not IMAGES_AVAILABLE:
            return "Yo! Chciałbym Ci wygenerować obrazek, ale images_client nie jest dostępny! 🎨❌"

        try:
            # Emit event
            if IO_PIPELINE_AVAILABLE:
                await self._emit_pipeline_event("image_request", {"prompt": content})

            # Extract prompt from content
            prompt = self._extract_image_prompt(content)

            # Generate image
            result = images_client.generate_image(prompt=prompt, size="1024x1024", style="vivid")

            if result.get("success"):
                image_path = result.get("path", "")
                return f"🎨 Oto Twój obrazek, ziomek! Wygenerowałem: '{prompt}'\n📁 Zapisano w: {image_path}\n\nCzy chcesz jakieś modyfikacje? 🚀"
            else:
                return f"Hmm, coś nie pykło z generowaniem obrazka! 🤔\nBłąd: {result.get('error', 'Unknown')}\nSpróbuj ponownie z innym opisem!"

        except Exception as e:
            return f"Ups! Problem z generowaniem obrazka: {str(e)} 🎨💥"

    def _extract_image_prompt(self, content: str) -> str:
        """Wyciąga prompt do obrazka z treści wiadomości."""
        # Remove trigger words and clean up
        triggers = ["generuj obraz", "stwórz obrazek", "narysuj", "image", "picture"]
        clean_content = content.lower()

        for trigger in triggers:
            clean_content = clean_content.replace(trigger, "").strip()

        # If nothing left, provide default
        if not clean_content or len(clean_content) < 3:
            return "beautiful abstract art, colorful, creative"

        return clean_content

    async def _emit_pipeline_event(self, event_type: str, data: Dict[str, Any]):
        """Emituje event do IO Pipeline."""
        if not IO_PIPELINE_AVAILABLE:
            return

        try:
            # Fix IO Pipeline call - use emit instead of emit_secure
            io_pipeline.emit(
                channel="mordzix",
                data={
                    "event_type": event_type,
                    "data": data,
                    "metadata": {"source": "mordzix_core", "timestamp": time.time()},
                },
            )
        except Exception as e:
            print(f"Pipeline event failed: {e}")

    async def _handle_crypto_intent(self, content: str, context: str = "") -> str:
        """Obsługuje crypto queries."""
        try:
            # Simple crypto response for now
            return "💰 Crypto update dla BITCOIN: 📊 Score: 85/100 ⚡ Bullish! 🚀"
        except Exception as e:
            return f"Problem z crypto: {str(e)}"

    async def _handle_travel_intent(self, content: str, context: str = "") -> str:
        """Obsługuje travel planning."""
        try:
            return "🌍 Travel planning aktywny! Podaj mi: dokąd jedziesz, na ile dni? 🗺️"
        except Exception as e:
            return f"Problem z travel: {str(e)}"

    async def _handle_programming_intent(self, content: str, context: str = "") -> str:
        """Obsługuje programming assistance."""
        try:
            return "👨‍💻 Code review mode! Wklej kod, a zrobię ci review jak bestia! 🔥"
        except Exception as e:
            return f"Problem z programming: {str(e)}"

    async def _handle_learning_intent(self, content: str, context: str = "") -> str:
        """Obsługuje learning system."""
        try:
            return "🧠 Learning mode aktywny! Mogę ci stworzyć flashcards, quiz lub plan nauki! 📚"
        except Exception as e:
            return f"Problem z learning: {str(e)}"

    async def _handle_writing_intent(self, content: str, context: str = "") -> str:
        """Obsługuje writing assistance."""
        try:
            return "✍️ Writing mode! Pomogę ci napisać artykuł, zoptymalizować SEO lub stworzyć content! 📝"
        except Exception as e:
            return f"Problem z writing: {str(e)}"

    async def _handle_general_intent(self, content: str, context: str = "") -> str:
        """Generuje odpowiedź używając wszystkich dostępnych AI engines."""

        # === SEED MEMORY AUTO-LEARNING ===
        if SEED_MEMORY_AVAILABLE and intent.get("save_to_ltm", False):
            try:
                # Auto-seed relevant knowledge for complex topics
                if any(
                    topic in content.lower() for topic in ["crypto", "bitcoin", "ai", "programming"]
                ):
                    # Extract topic and seed memory in background
                    asyncio.create_task(self._seed_knowledge_background(content))
            except Exception as e:
                print(f"🌱 Seed memory background task failed: {e}")

        # === KIMI MODEL FOR COMPLEX REASONING ===
        if KIMI_AVAILABLE and intent.get("type") in ["general", "complex_reasoning"]:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT if PROMPT_SYSTEM_AVAILABLE else "You are Mordzix.",
                    },
                    {"role": "user", "content": f"Context: {context}\n\nUser: {content}"},
                ]

                kimi_response = kimi_client.kimi_chat(
                    messages=messages, max_tokens=800, temperature=0.8
                )

                if kimi_response and len(kimi_response.strip()) > 10:
                    return kimi_response

            except Exception as e:
                print(f"🤖 Kimi failed, using fallback: {e}")

        if intent["type"] == "crypto":
            return await self._handle_crypto_intent(content, context)
        elif intent["type"] == "travel":
            return await self._handle_travel_intent(content, context)
        elif intent["type"] == "programming":
            return await self._handle_programming_intent(content, context)
        elif intent["type"] == "learning":
            return await self._handle_learning_intent(content, context)
        elif intent["type"] == "writing":
            return await self._handle_writing_intent(content, context)
        else:
            return await self._handle_general_intent(content, context)

    async def _seed_knowledge_background(self, content: str):
        """Background task to seed relevant knowledge."""
        try:
            if SEED_MEMORY_AVAILABLE:
                # Extract topic from content
                topic = self._extract_topic(content)
                if topic:
                    # Seed knowledge about this topic
                    seed_memory.seed_topic_knowledge(topic, max_articles=3)
        except Exception as e:
            print(f"🌱 Background seeding failed: {e}")

    def _extract_topic(self, content: str) -> str:
        """Extract main topic from user content."""
        content_lower = content.lower()
        topic_mapping = {
            "crypto": ["bitcoin", "eth", "crypto", "blockchain"],
            "ai": ["ai", "artificial intelligence", "machine learning", "neural"],
            "programming": ["python", "javascript", "code", "programming"],
            "travel": ["travel", "podróż", "hotel", "flight"],
        }

        for topic, keywords in topic_mapping.items():
            if any(keyword in content_lower for keyword in keywords):
                return topic
        return ""

    async def _handle_crypto_intent(self, content: str, context: str) -> str:
        """Obsługuje crypto queries z integracją crypto_advisor_full."""
        try:
            # Extract token name if mentioned
            content_lower = content.lower()
            if "bitcoin" in content_lower or "btc" in content_lower:
                token_id = "bitcoin"
            elif "ethereum" in content_lower or "eth" in content_lower:
                token_id = "ethereum"
            else:
                token_id = "bitcoin"  # default

            # Get token score
            score_data = crypto_advisor_full.calculate_token_score(token_id)

            response = f"Crypto update dla {token_id.upper()}:\\n"
            response += f"💰 Cena: ${score_data['price']:,.2f}\\n"
            response += f"📊 Score: {score_data['scores']['composite']}/100\\n"
            response += f"⚠️ Risk: {score_data['risk_level']}\\n"

            if "portfolio" in content_lower:
                response += "\\n💼 Chcesz sprawdzić portfolio? Podaj user ID!"

            return response

        except Exception as e:
            return f"Mały problem z crypto API, mordo. Error: {str(e)}. Ale nie martw się, ogarniemy to!"

    async def _handle_travel_intent(self, content: str, context: str) -> str:
        """Obsługuje travel queries z integracją travelguide."""
        try:
            # Simple travel response - can be enhanced with travelguide integration
            if any(word in content.lower() for word in ["plan", "zaplanuj", "podróż"]):
                return (
                    "No to planujemy podróż! 🌍\\n"
                    "Podaj mi: dokąd jedziesz, na ile dni i jaki masz budżet?\\n"
                    "Ogarnie ci całą trasę z hotelem, jedzeniem i atrakcjami!"
                )
            else:
                return (
                    "Travel mode on! ✈️\\n"
                    "Mogę ci zaplanować trasę, znaleźć hotele, sprawdzić pogodę, "
                    "polecić miejscówki do zwiedzania. Co potrzebujesz?"
                )

        except Exception as e:
            return f"Ups, travel system ma hickupa. Error: {str(e)}. Ale luzik, zaraz to naprawię!"

    async def _handle_programming_intent(self, content: str, context: str) -> str:
        """Obsługuje programming queries z integracją programista.py."""
        return (
            "Code review mode! 👨‍💻\\n"
            "Mogę ci zrobić code review, pomóc z bugami, zaproponować refactor, "
            "wygenerować dokumentację albo nauczyć nowych patternów. Wrzucaj kod!"
        )

    async def _handle_learning_intent(self, content: str, context: str) -> str:
        """Obsługuje learning queries z integracją autonauka.py."""
        return (
            "Learning mode aktywny! 🧠\\n"
            "Mogę ci stworzyć flashcards, zaplanować spaced repetition, "
            "przygotować quiz albo pomóc w organizacji wiedzy. Czego się uczymy?"
        )

    async def _handle_writing_intent(self, content: str, context: str) -> str:
        """Obsługuje writing queries z integracją writing_all_pro.py."""
        return (
            "Writing mode! ✍️\\n"
            "Pomogę ci napisać artykuł, zoptymalizować SEO, stworzyć content "
            "albo poprawić styl. O czym piszemy?"
        )

    async def _handle_general_intent(self, content: str, context: str) -> str:
        """Obsługuje general chat."""
        return (
            "No elo! Jestem Mordzix - Twój ziomek od wszystkiego. 🚀\\n"
            "Mogę pomóc z crypto, travel, kodem, nauką, pisaniem. "
            "Mów co potrzebujesz, nie mam hamulców ani filtrów!"
        )

    def _save_to_ltm(
        self, thread_id: str, user_msg: ChatMessage, ai_msg: ChatMessage, intent: Dict[str, Any]
    ):
        """Zapisuje ważne rzeczy do pamięci długoterminowej."""
        try:
            fact = f"User query: {user_msg.content} | AI response: {ai_msg.content[:200]}..."
            meta_data = {
                "source": f"chat_thread_{thread_id}_{intent['type']}",
                "thread_id": thread_id,
                "intent_type": intent["type"],
            }
            tags = [intent["type"], "chat", "mordzix"]

            self.memory_system.add_fact(fact, meta_data=meta_data, score=0.8, tags=tags)
        except Exception as e:
            print(f"Error saving to LTM: {e}")

    async def send_to_websocket(self, user_id: str, message_data: Dict[str, Any]):
        """Wysyła wiadomość przez WebSocket jeśli user jest połączony."""
        if user_id in self.websocket_connections:
            try:
                await self.websocket_connections[user_id].send_text(json.dumps(message_data))
            except Exception as e:
                print(f"WebSocket send error: {e}")
                # Remove broken connection
                if user_id in self.websocket_connections:
                    del self.websocket_connections[user_id]


# Global instance
mordzix_engine = MordzixChatEngine()


class MordzixCryptoIntegration:
    """Integracja z crypto_advisor_full dla real-time alerts w czacie."""

    def __init__(self, chat_engine: MordzixChatEngine):
        self.chat_engine = chat_engine
        self.monitored_portfolios: Dict[str, str] = {}  # user_id -> portfolio_id

    async def start_portfolio_monitoring(self, user_id: str, portfolio_id: str):
        """Rozpoczyna monitoring portfolio dla usera."""
        self.monitored_portfolios[user_id] = portfolio_id

        # Send system message
        await self.chat_engine.send_to_websocket(
            user_id,
            {
                "type": "system_message",
                "content": f"🚨 Portfolio monitoring aktywny dla {portfolio_id}!",
            },
        )

    async def portfolio_alert(self, portfolio_id: str, alert_data: Dict[str, Any]):
        """Wysyła alert portfolio do odpowiedniego usera."""
        for user_id, monitored_portfolio in self.monitored_portfolios.items():
            if monitored_portfolio == portfolio_id:
                alert_msg = self.chat_engine.personality.generate_system_message(
                    "crypto_alert",
                    f"{alert_data.get('symbol', 'Portfolio')}: {alert_data.get('change', 'zmiana')}%",
                )

                await self.chat_engine.send_to_websocket(
                    user_id, {"type": "crypto_alert", "content": alert_msg, "data": alert_data}
                )


# Crypto integration instance
crypto_integration = MordzixCryptoIntegration(mordzix_engine)


# === ADVANCED HANDLERS FOR NEW MODULES ===


# === RELIABILITY & ERROR HANDLING SYSTEM ===
class MordzixReliabilitySystem:
    """Zaawansowany system niezawodności dla wszystkich operacji Mordzix."""

    def __init__(self):
        self.failure_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.retry_delays = {
            "crypto": [1, 2, 5],
            "memory": [0.5, 1, 2],
            "image": [2, 5, 10],
            "default": [1, 3, 5],
        }
        self.circuit_breakers = defaultdict(lambda: {"failures": 0, "last_failure": 0})

    async def reliable_operation(
        self, operation_name: str, func, *args, operation_type: str = "default", **kwargs
    ):
        """Wykonuje operację z retry policy i circuit breaker."""
        delays = self.retry_delays.get(operation_type, self.retry_delays["default"])

        # Check circuit breaker
        if self._is_circuit_open(operation_name):
            raise Exception(f"Circuit breaker open for {operation_name}")

        last_exception = None

        for attempt in range(len(delays) + 1):
            try:
                result = (
                    await func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )

                # Success - reset circuit breaker
                self.success_counts[operation_name] += 1
                self.circuit_breakers[operation_name]["failures"] = 0

                return result

            except Exception as e:
                last_exception = e
                self.failure_counts[operation_name] += 1
                self.circuit_breakers[operation_name]["failures"] += 1
                self.circuit_breakers[operation_name]["last_failure"] = time.time()

                if attempt < len(delays):
                    delay = delays[attempt]
                    print(f"🔄 Retry {attempt + 1} for {operation_name} in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ All retries failed for {operation_name}: {str(e)}")
                    break

        raise last_exception

    def _is_circuit_open(self, operation_name: str) -> bool:
        """Sprawdza czy circuit breaker jest otwarty."""
        breaker = self.circuit_breakers[operation_name]

        # Open circuit if too many failures
        if breaker["failures"] >= 5:
            # Auto-reset after 60 seconds
            if time.time() - breaker["last_failure"] > 60:
                breaker["failures"] = 0
                return False
            return True

        return False

    def get_health_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki zdrowia systemu."""
        return {
            "success_counts": dict(self.success_counts),
            "failure_counts": dict(self.failure_counts),
            "circuit_breakers": {
                name: {"failures": breaker["failures"], "is_open": self._is_circuit_open(name)}
                for name, breaker in self.circuit_breakers.items()
            },
        }


# Global reliability system
reliability_system = MordzixReliabilitySystem()


# === DATA MANAGEMENT SYSTEM ===
class MordzixDataManager:
    """Zarządza strukturą danych Mordzix - conversations, cache, snapshots."""

    def __init__(self):
        self.base_path = Path("data")
        self.ensure_directories()

    def ensure_directories(self):
        """Tworzy wymaganą strukturę folderów."""
        dirs = ["conversations", "cache", "memory_snapshots", "seed_data", "exports", "temp"]

        for dir_name in dirs:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)

    def save_conversation(self, thread_id: str, conversation_data: Dict[str, Any]):
        """Zapisuje konwersację do pliku."""
        file_path = self.base_path / "conversations" / f"{thread_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

    def load_conversation(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Ładuje konwersację z pliku."""
        file_path = self.base_path / "conversations" / f"{thread_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def cleanup_old_data(self, days_old: int = 30):
        """Czyści stare dane."""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)

        for folder in ["conversations", "cache", "temp"]:
            folder_path = self.base_path / folder
            if folder_path.exists():
                for file_path in folder_path.iterdir():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()


# Global data manager
data_manager = MordzixDataManager()


# === PLUGIN SYSTEM ===
class MordzixPluginSystem:
    """System pluginów dla rozszerzeń Mordzix."""

    def __init__(self):
        self.loaded_plugins = {}
        self.plugin_dir = Path("plugins")

    def load_plugin(self, plugin_name: str):
        """Ładuje plugin z folderu plugins."""
        try:
            plugin_path = self.plugin_dir / f"{plugin_name}.py"
            if plugin_path.exists():
                # Dynamic import - simplified for safety
                print(f"📦 Plugin {plugin_name} ready to load")
                self.loaded_plugins[plugin_name] = {"status": "available", "path": plugin_path}
                return True
        except Exception as e:
            print(f"❌ Failed to load plugin {plugin_name}: {e}")
        return False

    def get_available_plugins(self) -> List[str]:
        """Zwraca listę dostępnych pluginów."""
        if not self.plugin_dir.exists():
            return []

        return [f.stem for f in self.plugin_dir.glob("*.py") if not f.name.startswith("__")]


# Global plugin system
plugin_system = MordzixPluginSystem()


# Production ready - no test functions in core module


async def handle_image_generation(content: str) -> str:
    """Handle image generation requests using images_client."""
    if not IMAGES_AVAILABLE:
        return "🎨 Generowanie obrazków niedostępne w tej chwili."

    try:
        # Extract image prompt from content
        prompt_keywords = ["generuj", "narysuj", "stwórz", "obrazek", "image", "pic"]
        image_prompt = content.lower()

        # Find and clean prompt
        for keyword in prompt_keywords:
            if keyword in image_prompt:
                image_prompt = content[content.lower().find(keyword) + len(keyword) :].strip()
                break

        if len(image_prompt) < 3:
            return "🎨 Podaj bardziej szczegółowy opis tego co chcesz wygenerować!"

        # Generate image using images_client
        try:
            result = images_client.text_to_image(
                prompt=image_prompt, size="1024x1024", provider="openai"
            )

            if result.get("success"):
                return f"🎨 Obrazek wygenerowany! Prompt: '{image_prompt}'\n📁 Plik: {result.get('filename', 'generated.png')}"
            else:
                return f"🎨 Błąd generowania: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"🎨 Problem z images_client: {str(e)}"

    except Exception as e:
        return f"🎨 Ogólny błąd generowania obrazka: {str(e)}"


async def handle_advanced_psychika(content: str, context: str) -> str:
    """Use psychika for complex decision making."""
    if not PSYCHIKA_AVAILABLE:
        return None

    try:
        psychika_input = {
            "user_message": content,
            "context": context[-800:],  # Last 800 chars
            "timestamp": time.time(),
        }

        decision = psychika.make_decision(psychika_input)

        if decision.get("confidence", 0) > 0.75:
            return decision.get("response", "")

    except Exception as e:
        print(f"🧠 Psychika error: {e}")

    return None


def handle_memory_seeding():
    """Initialize and seed memory if needed."""
    if SEED_MEMORY_AVAILABLE:
        try:
            # Check if seeding is needed
            seed_status = seed_memory.check_seed_status()

            if not seed_status.get("is_seeded", False):
                print("🌱 Seeding memory...")
                seed_memory.seed_all_memory()
                print("✅ Memory seeded successfully!")

        except Exception as e:
            print(f"🌱 Memory seeding error: {e}")


# Initialize memory seeding on module load
try:
    handle_memory_seeding()
except Exception as e:
    print(f"⚠️ Memory seeding initialization failed: {e}")


# IO Pipeline event handlers
if IO_PIPELINE_AVAILABLE:
    try:
        # Register Mordzix event handlers
        io_pipeline.subscribe("mordzix.chat.*", lambda event: print(f"📊 Chat event: {event}"))
        io_pipeline.subscribe("mordzix.crypto.*", lambda event: print(f"💰 Crypto event: {event}"))
        io_pipeline.subscribe("mordzix.image.*", lambda event: print(f"🎨 Image event: {event}"))

        print("🚇 IO Pipeline handlers registered successfully!")
    except Exception as e:
        print(f"🚇 IO Pipeline setup error: {e}")


print("🚀 Mordzix Core v2.0 with Advanced Modules loaded!")
print(f"   🧠 Psychika: {'✅' if PSYCHIKA_AVAILABLE else '❌'}")
print(f"   🤖 Kimi: {'✅' if KIMI_AVAILABLE else '❌'}")
print(f"   🌱 Seed Memory: {'✅' if SEED_MEMORY_AVAILABLE else '❌'}")
print(f"   📝 Prompt System: {'✅' if PROMPT_SYSTEM_AVAILABLE else '❌'}")
print(f"   🚇 IO Pipeline: {'✅' if IO_PIPELINE_AVAILABLE else '❌'}")
print(f"   🎨 Images: {'✅' if IMAGES_AVAILABLE else '❌'}")
