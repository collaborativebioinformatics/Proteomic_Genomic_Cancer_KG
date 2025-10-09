
---
# KG-HUB
---

As explained in: [Caufield et al. - 2023 - KG-Hub—building and exchanging biological knowledge graphs.pdf](file:///Users/andrew/Zotero/storage/P6MIW8M3/Caufield%20et%20al.%20-%202023%20-%20KG-Hub—building%20and%20exchanging%20biological%20knowledge%20graphs.pdf)

**KG-Hub is a collection of tools and libraries for building and reusing KGs**

---
## Data Representation

- All graphs in KG-Hub are represented as [directed, heterogeneous property graphs].
- [Edges] and [nodes] are typed according to a data model, edges have direction (e.g. A “affects risk for” B is distinct from B “affects risk for” A), and both nodes and edges may have one or more properties (e.g. a node may have a name or a textual description, and an edge may have a reference to a paper that provides provenance).
- An [association] minimally includes a subject and an object related by a Biolink Model predicate, together comprising its core triple (statement or primary assertion)

---
## How?

To unify the representation of overlapping concepts, terms, and data structures in these often disparate sources, KG-Hub uses a data model, [Biolink Model](Biolink%20Model.md) to allow cross-source interoperation.
### Processing raw input data into [KGX](KGX.md) [TSV format](TSV%20format.md)

Using [KGX](KGX.md) and [Koza](Koza.md) to transform data sources into standalone [Biolink Model](Biolink%20Model.md) compliant graphs. The transformation is done using a download.yaml configuration file and a declarative transformation configuration (enabled by [Koza](Koza.md)), and a Python control script. ([Example](https://github.com/Knowledge-Graph-Hub/kg-example))

1.  `download.yaml` - Documents and manages retrieval of data
2. `declarative transformation configuration` - Used by [Koza](Koza.md), and helps to transform the data
	- [Documentation](https://github.com/monarch-initiative/koza)
3. The final step in each graph’s ETL process is to merge the individual transform products into one final graph
	- Handled by `KGX merge` and defined in `merge.yaml`
	- Once transformed, each subgraph and the merged graph are available from a centralized open data repository (in Amazon Web Services, AWS), allowing users to reuse, mix, and match subgraphs, as well as share the final merged graph.

