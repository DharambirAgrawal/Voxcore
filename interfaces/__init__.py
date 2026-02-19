"""
VoxCore — interfaces module
User-facing interfaces: CLI debug, WebSocket real-time, REST API.
"""
from interfaces.cli import CLIInterface
from interfaces.websocket_server import WebSocketServer
from interfaces.api import APIServer

__all__ = ["CLIInterface", "WebSocketServer", "APIServer"]
