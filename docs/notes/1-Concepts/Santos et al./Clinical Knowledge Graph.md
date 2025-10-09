---
aliases:
  - CKG
---
# Clinical Knowledge Graph (CKG)

[Santos et al. - 2022 - A knowledge graph to interpret clinical proteomics data.pdf](file:///Users/andrew/Zotero/storage/GJFJZL6B/Santos%20et%20al.%20-%202022%20-%20A%20knowledge%20graph%20to%20interpret%20clinical%20proteomics%20data.pdf)

Clinical Knowledge Graph (CKG), an open-source platform currently comprising close to 20 million nodes and 220 million relationships that represent relevant experimental data, public databases and literature

- harmonization of proteomics with other omics data while integrating the relevant biomedical databases and **text extracted from scientific publications**
## Architecture Overview

![Pasted image 20251004140737](Pasted%20image%2020251004140737.png)
### analytics_core
- Contains the main steps in a data science pipeline:
	- **Data Preparation**: Filtering, normalization, imputation, and data formatting
	- **Data Exploration**: Summary statistics, ranking and distributions
	- **Data Analysis**: Dimensionality reduction, hypothesis testing, and correlations
	- **Visualization**: basic plots (e.g. bar plot) and complex plots (e.g. network, Sankey, or polar plots) using *Plot.ly*
- Integrates other data types in addition to proteomics:
	- Clinical data, multi-omics, biological context and text mining
- Some functionality in R:
	- SMAR and WGCNA
### graphdb_builder 
- Uses a [Neo4j](Neo4j) backend and creates a [Knowledge Graph](Knowledge%20Graph.md) with associated configurations for each ontology, database, and type of experiment.
- Integrates data from publicly accessible databases, user-conducted experiments, existing ontologies, and scientific publications
### graphdb_connector
Connects and queries database
### report_manager
Data visualization