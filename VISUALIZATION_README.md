# 🎨 Knowledge Graph Visualization Guide

This guide explains how to generate and view visualizations of your cancer knowledge graph.

## 📊 Two Types of Visualizations

### 1. 📈 Static Reports (PNG files)
**Location:** `reports/` folder  
**Purpose:** Professional, publication-ready figures for presentations and analysis

### 2. 🌐 Interactive Graph (HTML file)
**Location:** `data/output/interactive_knowledge_graph.html`  
**Purpose:** Explore the graph dynamically with hover, zoom, and filtering

---

## 🚀 Quick Start

### Generate Static Reports
```bash
uv run python src/generate_reports.py
```
This creates 5 numbered PNG files in the `reports/` folder.

### Create & Open Interactive Graph
```bash
uv run python src/interactive_graph.py
```
This creates the HTML file and opens it in your default browser.

### Just Open Existing Interactive Graph
```bash
uv run python open_interactive_graph.py
```
Opens the interactive graph if it already exists.

---

## 📁 Static Reports Overview

### `01_kg_overview.png`
- **What:** Overall graph structure and patient distributions
- **Use:** Understanding the dataset composition and graph statistics
- **Key insight:** Shows the balance of node types and patient characteristics

### `02_key_proteins_network.png`
- **What:** Network of key proteins from Zhang et al. paper and their patient connections
- **Use:** Understanding which patients connect to important biological proteins
- **Key insight:** Shows how MSI-H vs MSS patients relate to key proteins

### `03_msi_patients_network.png` 
- **What:** MSI-H patients and their top connected proteins, colored by subtype
- **Use:** Visualizing the key finding - different protein patterns for different subtypes
- **Key insight:** 🎯 **ZHANG'S KEY FINDING** - Shows why subtype B (green, good prognosis) vs C (red, poor prognosis) have different protein networks

### `04_protein_analysis.png`
- **What:** Analysis of protein categories, variance, and connectivity patterns
- **Use:** Understanding the biological categories of proteins in the graph
- **Key insight:** Distribution of MMR, EMT, adhesion, and other key protein categories

### `05_msi_subtype_analysis.png` 
- **What:** 🎯 **THE KEY FINDING** - MSI-H patients split by proteomic subtypes
- **Use:** **Core result for your hackathon presentation**
- **Key insight:** Shows how proteomics distinguishes what genomics lumps together

---

## 🌐 Interactive Graph Features

When you open the interactive graph, you can:

### 🎯 Node Types & Colors
- 🔴 **Red:** MSI-H patients (your classification targets)
- 🟢 **Green:** Key proteins (MLH1, CDH1, VIM, etc.)
- 🟠 **Orange:** Biological pathways (DNA repair, EMT, etc.)  
- 🟣 **Purple:** Proteomic subtypes (A, B, C, D, E)
- 🔵 **Blue:** Key mutations (BRAF, KRAS, TP53, POLE)

### 💡 Interactive Features
- **Hover:** See node details and connection counts
- **Drag:** Move nodes to explore connections
- **Zoom:** Mouse wheel to zoom in/out
- **Legend:** Click to hide/show node types
- **Layout:** Nodes are positioned to show relationships

### 🔍 What to Look For
1. **MSI-H patients** (red nodes) with different connection patterns
2. **Protein clusters** around different patient subtypes
3. **Pathway connections** to understand biological mechanisms
4. **Mutation patterns** distinguishing subtypes

---

## 📋 File Organization

```
Proteomic_Genomic_Cancer_KG/
├── reports/                          # 📊 Static visualization reports
│   ├── 01_kg_overview.png           # Graph structure overview  
│   ├── 02_key_proteins_network.png  # Key proteins network
│   ├── 03_msi_patients_network.png  # MSI-H patients network
│   ├── 04_protein_analysis.png      # Protein categories analysis
│   └── 05_msi_subtype_analysis.png  # 🎯 KEY FINDING visualization
│
├── data/output/
│   └── interactive_knowledge_graph.html  # 🌐 Interactive graph
│
├── src/
│   ├── generate_reports.py          # Generate all static reports
│   ├── interactive_graph.py         # Create interactive visualization
│   └── ...
│
└── open_interactive_graph.py        # Quick open interactive graph
```

---

## 🎯 For Your Hackathon Presentation

### Key Visualizations to Show:

1. **`05_msi_subtype_analysis.png`** - THE main finding
   - Shows MSI-H patients split into B (good) vs C (poor) prognosis
   - Demonstrates Zhang et al's key result

2. **`03_msi_patients_network.png`** - Network explanation  
   - Shows WHY proteomics distinguishes what genomics lumps together
   - Different protein connection patterns for different subtypes

3. **Interactive graph** - Live demonstration
   - Let audience explore the connections
   - Show how patients cluster by protein patterns

### Key Message:
> "Genomics tells us these are all MSI-H patients, but proteomics reveals two distinct subgroups with different clinical outcomes. Our knowledge graph shows the protein networks that distinguish good prognosis (Subtype B) from poor prognosis (Subtype C) patients."

---

## 🔧 Regenerating Visualizations

If you update your data or want to refresh the visualizations:

```bash
# Regenerate all static reports
uv run python src/generate_reports.py

# Regenerate interactive graph  
uv run python src/interactive_graph.py
```

The scripts are designed to overwrite existing files, so you can run them multiple times safely.

---

## 📊 Next Steps

These visualizations provide the foundation for:
1. **Understanding your data** - Patient and protein distributions
2. **Explaining Zhang's finding** - Why proteomics matters
3. **Building your GNN** - Node and edge structure is clear
4. **Presenting results** - Professional figures ready to use

Your knowledge graph perfectly captures the biological relationships needed to replicate and extend Zhang et al's groundbreaking work! 🚀