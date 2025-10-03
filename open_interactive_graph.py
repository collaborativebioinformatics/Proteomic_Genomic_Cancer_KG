#!/usr/bin/env python3
"""
Simple script to open the interactive knowledge graph in the default browser.
"""
import webbrowser
import os
from pathlib import Path

def open_interactive_graph():
    """Open the interactive knowledge graph in the default browser."""
    
    # Path to the interactive HTML file
    html_file = Path("data/output/interactive_knowledge_graph.html")
    
    if html_file.exists():
        # Convert to absolute path for browser
        abs_path = html_file.absolute()
        file_url = f"file://{abs_path}"
        
        print(f"🌐 Opening interactive knowledge graph...")
        print(f"📂 File: {html_file}")
        print(f"🔗 URL: {file_url}")
        
        # Open in default browser
        webbrowser.open(file_url)
        
        print("✅ Interactive graph opened in your browser!")
        print("\n🎯 What you'll see:")
        print("  🔴 Red nodes: MSI-H patients (your key targets)")
        print("  🟢 Green nodes: Key proteins (MMR, EMT, etc.)")
        print("  🟠 Orange nodes: Biological pathways")
        print("  🟣 Purple nodes: Proteomic subtypes (B, C, etc.)")
        print("  🔵 Blue nodes: Mutations (BRAF, KRAS, etc.)")
        print("\n💡 Interactive features:")
        print("  • Hover over nodes for details")
        print("  • Click and drag to explore")
        print("  • Zoom in/out with mouse wheel")
        print("  • Click legend to hide/show node types")
        
    else:
        print(f"❌ Interactive graph file not found: {html_file}")
        print("🔧 Create it first with:")
        print("   uv run python src/interactive_graph.py")

if __name__ == "__main__":
    open_interactive_graph()