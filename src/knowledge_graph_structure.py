import pandas as pd
import numpy as np
from loguru import logger
import networkx as nx
from typing import Dict, List, Tuple, Set
import json

class CancerKnowledgeGraph:
    """
    Knowledge Graph structure for cancer subtype classification.
    
    Replicates Zhang et al's key finding: proteomics splits MSI/CIMP subtype 
    into two distinct groups (Subtype B and C) with different clinical outcomes.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_types = {
            'patient': [],
            'protein': [], 
            'pathway': [],
            'mutation': [],
            'subtype': []
        }
        self.edge_types = {
            'patient_protein': [],      # Patient -> Protein abundance
            'protein_pathway': [],      # Protein -> Pathway membership
            'patient_mutation': [],     # Patient -> Mutation status
            'patient_subtype': [],      # Patient -> Proteomic subtype
            'protein_protein': []       # Protein -> Protein correlations
        }
    
    def create_graph_structure(self):
        """
        Create the heterogeneous knowledge graph structure.
        """
        logger.info("=== CREATING KNOWLEDGE GRAPH STRUCTURE ===")
        
        # Load processed datasets
        metadata_df = pd.read_csv('data/output/kg_patient_metadata.csv')
        protein_df = pd.read_csv('data/output/kg_protein_abundance.csv', index_col=0)
        protein_info_df = pd.read_csv('data/output/kg_protein_info.csv')
        
        logger.info(f"Loaded data: {len(metadata_df)} patients, {len(protein_df)} proteins")
        
        # Step 1: Create Patient Nodes
        self._create_patient_nodes(metadata_df)
        
        # Step 2: Create Protein Nodes  
        self._create_protein_nodes(protein_df, protein_info_df)
        
        # Step 3: Create Pathway Nodes
        self._create_pathway_nodes()
        
        # Step 4: Create Mutation & Subtype Nodes
        self._create_mutation_subtype_nodes(metadata_df)
        
        # Step 5: Create Edges
        self._create_edges(metadata_df, protein_df)
        
        # Step 6: Add to NetworkX graph
        self._build_networkx_graph()
        
        logger.info(f"Final KG: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _create_patient_nodes(self, metadata_df: pd.DataFrame):
        """Create patient nodes with clinical/genomic features."""
        logger.info("Creating patient nodes...")
        
        for _, patient in metadata_df.iterrows():
            patient_id = f"patient_{patient['patient_id']}"
            
            # Patient features for GNN
            features = {
                'node_type': 'patient',
                'tcga_id': patient['TCGA participant ID'],
                'msi_status': patient['MSI.status'],
                'proteomic_subtype': patient['Proteomic.subtype'],
                'age': patient['age_at_initial_pathologic_diagnosis'],
                'stage': patient['stage_numeric'],
                'gender': patient['Gender'],
                'tumor_site': patient['Tumor.site'],
                'braf_mut': patient['BRAF mutation'],
                'kras_mut': patient['KRAS mutation'], 
                'tp53_mut': patient['TP53 mutation'],
                'vital_status': patient['vital_status'],
                # Target labels
                'is_msi_h': 1 if patient['MSI.status'] == 'MSI-H' else 0,
                'is_subtype_b': patient['is_subtype_B'],
                'is_subtype_c': patient['is_subtype_C']
            }
            
            self.node_types['patient'].append((patient_id, features))
    
    def _create_protein_nodes(self, protein_df: pd.DataFrame, protein_info_df: pd.DataFrame):
        """Create protein nodes with biological annotations."""
        logger.info("Creating protein nodes...")
        
        for protein in protein_df.index:
            protein_id = f"protein_{protein}"
            protein_info = protein_info_df[protein_info_df['protein'] == protein].iloc[0]
            
            features = {
                'node_type': 'protein',
                'gene_symbol': protein,
                'is_key_protein': protein_info['is_key_protein'],
                'variance': protein_info['variance'],
                'protein_category': self._get_protein_category(protein)
            }
            
            self.node_types['protein'].append((protein_id, features))
    
    def _create_pathway_nodes(self):
        """Create pathway nodes based on Zhang et al. key pathways."""
        logger.info("Creating pathway nodes...")
        
        # Key pathways from Zhang paper
        pathways = {
            'DNA_Mismatch_Repair': {
                'proteins': ['MLH1', 'MSH2', 'MSH6', 'PMS2'],
                'description': 'DNA mismatch repair pathway - key for MSI status'
            },
            'Cell_Adhesion': {
                'proteins': ['CDH1', 'CTNNB1', 'CTNNA1', 'DSG2', 'PKP2'],
                'description': 'Cell adhesion/E-cadherin complex - distinguishes subtypes B vs C'
            },
            'ECM_Organization': {
                'proteins': ['COL1A1', 'COL3A1', 'FN1', 'VIM'],
                'description': 'Extracellular matrix - high in poor prognosis Subtype C'
            },
            'Cell_Proliferation': {
                'proteins': ['MKI67', 'PCNA'],
                'description': 'Cell cycle and proliferation markers'
            },
            'Immune_Response': {
                'proteins': ['CD3E', 'CD8A'], 
                'description': 'Immune infiltration markers'
            }
        }
        
        for pathway_name, info in pathways.items():
            pathway_id = f"pathway_{pathway_name}"
            features = {
                'node_type': 'pathway',
                'pathway_name': pathway_name,
                'description': info['description'],
                'proteins': info['proteins']
            }
            self.node_types['pathway'].append((pathway_id, features))
    
    def _create_mutation_subtype_nodes(self, metadata_df: pd.DataFrame):
        """Create mutation and subtype nodes."""
        logger.info("Creating mutation and subtype nodes...")
        
        # Mutation nodes
        mutations = ['BRAF', 'KRAS', 'TP53', 'POLE']
        for mut in mutations:
            mut_id = f"mutation_{mut}"
            features = {
                'node_type': 'mutation',
                'gene': mut,
                'mutation_type': 'point_mutation'
            }
            self.node_types['mutation'].append((mut_id, features))
        
        # Proteomic subtype nodes
        subtypes = ['A', 'B', 'C', 'D', 'E']
        for subtype in subtypes:
            subtype_id = f"subtype_{subtype}"
            features = {
                'node_type': 'subtype',
                'subtype': subtype,
                'clinical_outcome': self._get_subtype_outcome(subtype)
            }
            self.node_types['subtype'].append((subtype_id, features))
    
    def _create_edges(self, metadata_df: pd.DataFrame, protein_df: pd.DataFrame):
        """Create edges between nodes."""
        logger.info("Creating edges...")
        
        # 1. Patient -> Protein edges (abundance values)
        self._create_patient_protein_edges(metadata_df, protein_df)
        
        # 2. Protein -> Pathway edges
        self._create_protein_pathway_edges()
        
        # 3. Patient -> Mutation edges
        self._create_patient_mutation_edges(metadata_df)
        
        # 4. Patient -> Subtype edges
        self._create_patient_subtype_edges(metadata_df)
        
        # 5. Protein -> Protein correlations (optional, for dense connections)
        self._create_protein_correlations(protein_df)
    
    def _create_patient_protein_edges(self, metadata_df: pd.DataFrame, protein_df: pd.DataFrame):
        """Create patient-protein edges with abundance as edge weight."""
        logger.info("Creating patient-protein edges...")
        
        for _, patient in metadata_df.iterrows():
            patient_id = f"patient_{patient['patient_id']}"
            tcga_id = patient['TCGA participant ID']
            
            if tcga_id in protein_df.columns:
                for protein in protein_df.index:
                    protein_id = f"protein_{protein}"
                    abundance = protein_df.loc[protein, tcga_id]
                    
                    # Only create edge if abundance > threshold (reduce graph density)
                    if abundance > 2.0:  # Log2-transformed values
                        edge = (patient_id, protein_id, {
                            'edge_type': 'patient_protein',
                            'abundance': abundance,
                            'weight': abundance / 10.0  # Normalize for GNN
                        })
                        self.edge_types['patient_protein'].append(edge)
    
    def _create_protein_pathway_edges(self):
        """Create protein-pathway membership edges."""
        logger.info("Creating protein-pathway edges...")
        
        for pathway_id, pathway_features in self.node_types['pathway']:
            pathway_proteins = pathway_features['proteins']
            
            for protein_symbol in pathway_proteins:
                protein_id = f"protein_{protein_symbol}"
                # Check if protein exists in our filtered set
                if any(protein_id == pid for pid, _ in self.node_types['protein']):
                    edge = (protein_id, pathway_id, {
                        'edge_type': 'protein_pathway',
                        'weight': 1.0
                    })
                    self.edge_types['protein_pathway'].append(edge)
    
    def _create_patient_mutation_edges(self, metadata_df: pd.DataFrame):
        """Create patient-mutation edges."""
        logger.info("Creating patient-mutation edges...")
        
        mutation_cols = {
            'BRAF mutation': 'BRAF',
            'KRAS mutation': 'KRAS', 
            'TP53 mutation': 'TP53',
            'POLE mutation': 'POLE'
        }
        
        for _, patient in metadata_df.iterrows():
            patient_id = f"patient_{patient['patient_id']}"
            
            for col, mut_name in mutation_cols.items():
                if patient[col] == 1.0:  # Mutation present
                    mut_id = f"mutation_{mut_name}"
                    edge = (patient_id, mut_id, {
                        'edge_type': 'patient_mutation',
                        'weight': 1.0
                    })
                    self.edge_types['patient_mutation'].append(edge)
    
    def _create_patient_subtype_edges(self, metadata_df: pd.DataFrame):
        """Create patient-subtype edges."""
        logger.info("Creating patient-subtype edges...")
        
        for _, patient in metadata_df.iterrows():
            patient_id = f"patient_{patient['patient_id']}"
            subtype = patient['Proteomic.subtype']
            
            if pd.notna(subtype):
                subtype_id = f"subtype_{subtype}"
                edge = (patient_id, subtype_id, {
                    'edge_type': 'patient_subtype',
                    'weight': 1.0
                })
                self.edge_types['patient_subtype'].append(edge)
    
    def _create_protein_correlations(self, protein_df: pd.DataFrame, correlation_threshold=0.7):
        """Create protein-protein correlation edges (optional - can make graph very dense)."""
        logger.info("Creating protein-protein correlation edges...")
        
        # Calculate correlations
        corr_matrix = protein_df.T.corr()  # Transpose for patient correlations
        
        # Only add strong correlations to avoid too many edges
        proteins = protein_df.index.tolist()
        for i, prot1 in enumerate(proteins):
            for j, prot2 in enumerate(proteins[i+1:], i+1):
                correlation = corr_matrix.loc[prot1, prot2]
                
                if abs(correlation) > correlation_threshold:
                    prot1_id = f"protein_{prot1}"
                    prot2_id = f"protein_{prot2}"
                    
                    edge = (prot1_id, prot2_id, {
                        'edge_type': 'protein_protein',
                        'correlation': correlation,
                        'weight': abs(correlation)
                    })
                    self.edge_types['protein_protein'].append(edge)
    
    def _build_networkx_graph(self):
        """Build the final NetworkX graph."""
        logger.info("Building NetworkX graph...")
        
        # Add all nodes
        for node_type, nodes in self.node_types.items():
            for node_id, features in nodes:
                self.graph.add_node(node_id, **features)
        
        # Add all edges
        for edge_type, edges in self.edge_types.items():
            for edge in edges:
                if len(edge) == 3:
                    source, target, attributes = edge
                    self.graph.add_edge(source, target, **attributes)
        
        logger.info(f"Graph statistics:")
        for node_type in self.node_types:
            count = len([n for n, d in self.graph.nodes(data=True) if d['node_type'] == node_type])
            logger.info(f"  {node_type} nodes: {count}")
    
    def _get_protein_category(self, protein: str) -> str:
        """Get protein category based on biological function."""
        mmr_proteins = ['MLH1', 'MSH2', 'MSH6', 'PMS2']
        emt_markers = ['CDH1', 'VIM', 'COL1A1', 'COL3A1', 'FN1'] 
        cell_adhesion = ['CTNNB1', 'CTNNA1', 'DSG2', 'PKP2']
        proliferation = ['MKI67', 'PCNA']
        immune = ['CD3E', 'CD8A']
        
        if protein in mmr_proteins:
            return 'MMR'
        elif protein in emt_markers:
            return 'EMT'
        elif protein in cell_adhesion:
            return 'cell_adhesion'
        elif protein in proliferation:
            return 'proliferation'
        elif protein in immune:
            return 'immune'
        else:
            return 'other'
    
    def _get_subtype_outcome(self, subtype: str) -> str:
        """Get clinical outcome for proteomic subtypes based on Zhang paper."""
        outcomes = {
            'A': 'intermediate',
            'B': 'good',      # MSI-H + good prognosis  
            'C': 'poor',      # MSI-H + poor prognosis
            'D': 'intermediate',
            'E': 'poor'       # CIN + poor prognosis
        }
        return outcomes.get(subtype, 'unknown')
    
    def save_graph_structure(self, output_dir='data/output/'):
        """Save the graph structure and metadata."""
        logger.info("Saving graph structure...")
        
        # Convert numpy booleans to Python booleans for GraphML compatibility
        for node_id, attributes in self.graph.nodes(data=True):
            for key, value in attributes.items():
                if isinstance(value, np.bool_):
                    attributes[key] = bool(value)
                elif isinstance(value, np.integer):
                    attributes[key] = int(value)
                elif isinstance(value, np.floating):
                    attributes[key] = float(value)
        
        for edge in self.graph.edges(data=True):
            _, _, attributes = edge
            for key, value in attributes.items():
                if isinstance(value, np.bool_):
                    attributes[key] = bool(value)
                elif isinstance(value, np.integer):
                    attributes[key] = int(value)
                elif isinstance(value, np.floating):
                    attributes[key] = float(value)
        
        # Save as pickle (more reliable for Python objects)
        import pickle
        with open(f"{output_dir}cancer_knowledge_graph.pkl", 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save as GML (simpler text format)
        nx.write_gml(self.graph, f"{output_dir}cancer_knowledge_graph.gml")
        
        # Save node and edge statistics
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_counts': {
                node_type: len([n for n, d in self.graph.nodes(data=True) if d['node_type'] == node_type])
                for node_type in self.node_types
            },
            'edge_counts': {
                edge_type: len(edges) for edge_type, edges in self.edge_types.items()
            }
        }
        
        with open(f"{output_dir}kg_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("✅ Knowledge graph structure saved!")
        return stats

def main():
    """Main function to create the knowledge graph."""
    kg = CancerKnowledgeGraph()
    graph = kg.create_graph_structure()
    stats = kg.save_graph_structure()
    
    print("\n=== KNOWLEDGE GRAPH SUMMARY ===")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print("\nNode distribution:")
    for node_type, count in stats['node_counts'].items():
        print(f"  {node_type}: {count}")
    print("\nEdge distribution:")
    for edge_type, count in stats['edge_counts'].items():
        print(f"  {edge_type}: {count}")
    
    return kg, graph

if __name__ == "__main__":
    main()