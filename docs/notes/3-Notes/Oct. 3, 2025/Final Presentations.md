**General themes**
- Standardized data pipelines
- KGs to use with a RAG process
- Federated KGs

carnegie mellon x nvidia federated learning hackathon
- come prepared to normalize a lot of data
- october 31 paper release, dec 1st 8 am to 4 pm (ET)
	- online, hell yeah

[BioGraphRAG](https://github.com/collaborativebioinformatics/BioGraphRAG)
- **NextFlow** is very popular

[KG Model Garbage Collection](https://github.com/collaborativebioinformatics/Model_Garbage_Collection)
- Pruning false edges from a KG
- Given a set of assertions (edges), an existing LMM, classify them as true, synthetic data labeled as false, inductive reason scoring for edges
	- You use the existing KG to make it "learn itself"
	- You can then manually prune edges and recompute the scores!
- What is a way organizers can facilitate people coming back?

[MIDAS](https://github.com/collaborativebioinformatics/MIDAS)
- More data from cBioPortal can be integrated into a larger KG
- Used ORIN, subset of **ROBOKOP**, AWS Bedrock,  and Gremlin
- Exported to OpenCypher format
- Combines knowledge graphs in KGX format, using BIolink Model
- Potential future work for prompt engineering
- Node normalizer, ncat translator project, **Babel**
	- https://github.com/NCATSTranslator/NodeNormalization

[EasyGiraffe](https://github.com/collaborativebioinformatics/EasyGiraffe)
- Populations that need to disclose genetic background then you can collectedly extract variants based on that
- Using **VGToolkit**, assimilator that generates FASTQ file
- Also used **ROBOKOP** to get sequence variants related to disease
- Applications
	- New sample, does it contain pathogenetic variance that they care about... From there can you identify a quick treatment based on the genetic background of the patient
	- Getting you out of just one individual reference genome
	- When you add more gene connections they'll point to one each other "like a subway map"

[ECoGraph](https://github.com/collaborativebioinformatics/ECoGraph)
- TCGA knowledge graph for COAD cancer
- S3, Amazon bedrock, amazon neptune
- KG to use on GraphRAG

[GeNETwork](https://github.com/collaborativebioinformatics/GeNETwork)
- Combined various data sources using Neo4j in Docker as the graph database
- Linkages from drug and genetic variants, can a doctor use that info for a more personalized treatment?
- Pathways are sub pathways of pathways... More statistical power for including all associations?
	- Topology analysis can help, hallmark node, gene node, hm <-> gene
- Not all genes are expressed in every cell, will this make the data representation more meaningful or complex?
	- Single cell data, selectively overlap it with cells... Would be better to do that?!
	- CellByGene (open data), **Q-Graph**
- Sets of GTACS that are ready, federated 