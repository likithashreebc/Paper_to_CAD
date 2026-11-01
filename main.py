#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Paper2CAD — An Interactive AI-Based Schematic Inspector
================================================================================
Unified Single Entry Point for Paper2CAD.

Usage:
    python main.py               # Starts the Web Studio and auto-opens browser
    python main.py --port 8000   # Custom port
    python main.py --no-browser  # Headless / server mode
    python main.py --mcp         # Starts the official Model Context Protocol (MCP) server
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Automatically load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline"))
sys.path.insert(0, str(REPO_ROOT / "ui"))


def print_banner(port: int = 8000, mode: str = "web"):
    banner = f"""
================================================================================
  Paper2CAD: An Interactive AI-Based Schematic Inspector
================================================================================
  • Mode:              {"MCP Server (stdio)" if mode == "mcp" else "Web Studio Studio"}
  • Web URL:           http://localhost:{port}
  • AI Assistant:      Active (Grounded Topological Engine)
  • Rule Validator:    Active (Shorts, Floating Pins, Ground Reference)
  • CAD Engine:        Active (IEEE Standard SVG & Schemdraw)
================================================================================
"""
    print(banner)


def check_environment():
    """Validates presence of core assets and directories."""
    weights_dir = REPO_ROOT / "weights"
    yolo_weight = weights_dir / "yolo" / "best.pt"
    if not yolo_weight.exists():
        print(f"[WARN] YOLOv10 weights not found at {yolo_weight}. Run pipeline setup if necessary.")

    # Ensure output and jobs directories exist
    (REPO_ROOT / "outputs" / "ui_jobs").mkdir(parents=True, exist_ok=True)


def start_web_app(host: str = "127.0.0.1", port: int = 8000, auto_open: bool = True):
    """Starts the Paper2CAD Web Studio application."""
    if sys.platform == "win32":
        import asyncio
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    import uvicorn
    from server import app

    print_banner(port=port, mode="web")
    url = f"http://localhost:{port}"

    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print(f"[START] Web Studio active at: {url}")
    print("[INFO] Press CTRL+C to stop the server.\n")
    uvicorn.run(app, host=host, port=port, loop="asyncio")


def start_mcp_server():
    """Starts the official Paper2CAD MCP Server over stdio transport."""
    print_banner(mode="mcp")
    from mcp_server import mcp_server
    mcp_server.run()


def main():
    parser = argparse.ArgumentParser(
        description="Paper2CAD: An Interactive AI-Based Schematic Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python main.py\n  python main.py --port 8080\n  python main.py --mcp"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the web server (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser on launch")
    parser.add_argument("--mcp", action="store_true", help="Launch the official MCP (Model Context Protocol) server instead")

    args = parser.parse_args()

    check_environment()

    if args.mcp:
        start_mcp_server()
    else:
        start_web_app(host=args.host, port=args.port, auto_open=not args.no_browser)


if __name__ == "__main__":
    main()
