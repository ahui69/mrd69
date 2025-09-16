"""
runpod_sync.py - moduł do synchronizacji bazy danych z RunPod

Ten moduł umożliwia synchronizację lokalnej bazy danych z RunPod Storage.
Obsługuje automatyczne tworzenie kopii zapasowych oraz odtwarzanie bazy
z RunPod, co zapewnia ciągłość działania aplikacji po restarcie kontenera.
"""

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import requests

import config
from memory import _vacuum_if_needed

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("backend.log"), logging.StreamHandler()],
)

logger = logging.getLogger("runpod_sync")

# Stałe
SYNC_INTERVAL = int(os.getenv("RUNPOD_SYNC_INTERVAL", "1800"))  # 30 minut domyślnie
MAX_BACKUP_RETRIES = 3
RUNPOD_API_URL = "https://api.runpod.io/v2"


class RunPodSync:
    """Klasa obsługująca synchronizację z RunPod Storage."""

    def __init__(self):
        self.api_key = config.RUNPOD_API_KEY
        self.endpoint_id = config.RUNPOD_ENDPOINT_ID
        self.persist_dir = Path(config.RUNPOD_PERSIST_DIR)
        self.use_runpod = config.USE_RUNPOD
        self.local_db_path = Path(os.path.join(Path(__file__).parent, "data", "memory.db"))
        self.remote_db_path = self.persist_dir / "data" / "ltm.db"
        self.sync_thread = None
        self._stop_event = threading.Event()

    def _create_backup(self) -> bool:
        """Tworzy kopię zapasową lokalnej bazy danych w RunPod Storage."""
        if not self.use_runpod or not self.api_key:
            return False

        try:
            # Upewnij się, że katalog docelowy istnieje
            self.remote_db_path.parent.mkdir(parents=True, exist_ok=True)

            # Wykonaj VACUUM na bazie przed kopią
            _vacuum_if_needed(0)

            # Skopiuj plik bazy danych
            shutil.copy2(self.local_db_path, self.remote_db_path)

            # Skopiuj pliki WAL i SHM jeśli istnieją
            for suffix in ["-wal", "-shm"]:
                local_side_file = self.local_db_path.with_name(self.local_db_path.name + suffix)
                remote_side_file = self.remote_db_path.with_name(self.remote_db_path.name + suffix)

                if local_side_file.exists():
                    shutil.copy2(local_side_file, remote_side_file)

            logger.info(f"Kopia zapasowa bazy danych utworzona w RunPod: {self.remote_db_path}")
            return True

        except Exception as e:
            logger.error(f"Błąd podczas tworzenia kopii zapasowej: {str(e)}")
            return False

    def _restore_from_backup(self) -> bool:
        """Odtwarza bazę danych z kopii zapasowej w RunPod Storage."""
        if not self.use_runpod or not self.remote_db_path.exists():
            return False

        try:
            # Upewnij się, że katalog docelowy istnieje
            self.local_db_path.parent.mkdir(parents=True, exist_ok=True)

            # Skopiuj plik bazy danych
            shutil.copy2(self.remote_db_path, self.local_db_path)

            # Skopiuj pliki WAL i SHM jeśli istnieją
            for suffix in ["-wal", "-shm"]:
                remote_side_file = self.remote_db_path.with_name(self.remote_db_path.name + suffix)
                local_side_file = self.local_db_path.with_name(self.local_db_path.name + suffix)

                if remote_side_file.exists():
                    shutil.copy2(remote_side_file, local_side_file)

            logger.info(f"Baza danych odtworzona z kopii zapasowej RunPod: {self.remote_db_path}")
            return True

        except Exception as e:
            logger.error(f"Błąd podczas odtwarzania z kopii zapasowej: {str(e)}")
            return False

    def check_pod_status(self) -> dict[str, Any]:
        """Sprawdza status RunPod dla tego endpointu."""
        if not self.api_key or not self.endpoint_id:
            return {"status": "not_configured"}

        try:
            url = f"{RUNPOD_API_URL}/endpoint/{self.endpoint_id}"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Błąd API RunPod: {response.status_code} - {response.text}")
                return {"status": "error", "code": response.status_code, "message": response.text}

        except Exception as e:
            logger.error(f"Błąd podczas sprawdzania statusu RunPod: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _sync_worker(self):
        """Funkcja wykonywana w wątku synchronizacji."""
        logger.info("Uruchomiono wątek synchronizacji RunPod")

        while not self._stop_event.is_set():
            try:
                # Tworzenie kopii zapasowej
                success = self._create_backup()
                if success:
                    logger.info("Synchronizacja z RunPod zakończona pomyślnie")
                else:
                    logger.warning("Nie można utworzyć kopii zapasowej w RunPod")

                # Oczekiwanie na następną synchronizację
                self._stop_event.wait(SYNC_INTERVAL)
            except Exception as e:
                logger.error(f"Błąd w wątku synchronizacji: {str(e)}")
                # Krótsze oczekiwanie przy błędzie
                self._stop_event.wait(300)  # 5 minut

    def start_sync(self):
        """Rozpoczyna wątek synchronizacji."""
        if not self.use_runpod:
            logger.info("RunPod nie jest włączony, synchronizacja nie została uruchomiona")
            return

        if self.sync_thread is not None and self.sync_thread.is_alive():
            logger.warning("Wątek synchronizacji już działa")
            return

        # Resetujemy flagę zatrzymania
        self._stop_event.clear()

        # Sprawdź, czy możemy odtworzyć bazę z RunPod
        if self.remote_db_path.exists() and (
            not self.local_db_path.exists()
            or self.remote_db_path.stat().st_size > self.local_db_path.stat().st_size
        ):
            logger.info("Znaleziono bazę danych w RunPod, odtwarzanie...")
            self._restore_from_backup()

        # Uruchom wątek synchronizacji
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        logger.info("Uruchomiono synchronizację z RunPod")

    def stop_sync(self):
        """Zatrzymuje wątek synchronizacji."""
        if self.sync_thread is not None and self.sync_thread.is_alive():
            self._stop_event.set()
            self.sync_thread.join(timeout=5)
            logger.info("Zatrzymano synchronizację z RunPod")

    def force_sync(self) -> bool:
        """Wymusza natychmiastową synchronizację z RunPod."""
        if not self.use_runpod:
            return False

        return self._create_backup()


# Singleton dla synchronizacji RunPod
_RUNPOD_SYNC_INSTANCE: RunPodSync | None = None


def get_runpod_sync() -> RunPodSync:
    """Zwraca singleton klasy RunPodSync."""
    global _RUNPOD_SYNC_INSTANCE
    if _RUNPOD_SYNC_INSTANCE is None:
        _RUNPOD_SYNC_INSTANCE = RunPodSync()
    return _RUNPOD_SYNC_INSTANCE


# Funkcje pomocnicze do użycia w głównej aplikacji
def start_runpod_sync():
    """Rozpoczyna synchronizację z RunPod."""
    sync = get_runpod_sync()
    sync.start_sync()


def stop_runpod_sync():
    """Zatrzymuje synchronizację z RunPod."""
    sync = get_runpod_sync()
    sync.stop_sync()


def force_runpod_sync() -> bool:
    """Wymusza natychmiastową synchronizację z RunPod."""
    sync = get_runpod_sync()
    return sync.force_sync()


if __name__ == "__main__":
    # Testowanie funkcjonalności z linii poleceń
    import argparse

    parser = argparse.ArgumentParser(
        description="RunPod Sync - narzędzie do synchronizacji z RunPod"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("start", help="Rozpocznij synchronizację")
    subparsers.add_parser("stop", help="Zatrzymaj synchronizację")
    subparsers.add_parser("force", help="Wymuś natychmiastową synchronizację")
    subparsers.add_parser("status", help="Sprawdź status RunPod")
    subparsers.add_parser("restore", help="Odtwórz bazę danych z RunPod")

    args = parser.parse_args()
    sync = get_runpod_sync()

    if args.command == "start":
        sync.start_sync()
        print("Rozpoczęto synchronizację z RunPod")

    elif args.command == "stop":
        sync.stop_sync()
        print("Zatrzymano synchronizację z RunPod")

    elif args.command == "force":
        success = sync.force_sync()
        print(f"Synchronizacja {'zakończona pomyślnie' if success else 'nie powiodła się'}")

    elif args.command == "status":
        status = sync.check_pod_status()
        print(f"Status RunPod: {status}")

    elif args.command == "restore":
        success = sync._restore_from_backup()
        print(f"Odtworzenie {'zakończone pomyślnie' if success else 'nie powiodło się'}")

    else:
        parser.print_help()
