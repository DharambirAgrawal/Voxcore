"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          VOXCORE — core/__init__.py                             ║
║                       CORE PACKAGE — THE BRAIN STEM                            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Package initializer for the core module. Exports the three foundational
    classes that every other module depends on.

EXPORTS:
    - Session        (from core.session)
    - EventBus       (from core.event_bus)
    - TurnManager    (from core.turn_manager)

USAGE:
    from core import Session, EventBus, TurnManager
"""

from core.session import Session
from core.event_bus import EventBus
from core.turn_manager import TurnManager
