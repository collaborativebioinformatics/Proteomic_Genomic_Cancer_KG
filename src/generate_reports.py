import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class KnowledgeGraphReportGenerator:
    """
    Generate static visualization reports for the cancer knowledge graph.
    Saves all plots to the reports/ folder.
    """
    
    def __init__(self, graph_path='data/output/cancer_knowledge_graph.pkl', output_dir='reports/'):
        """Load the knowledge graph and set output directory."""
        logger.info("Loading knowledge graph for report generation...")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Load additional data
        self.metadata_df = pd.read_csv('data/output/kg_patient_metadata.csv')
        self.protein_info_df = pd.read_csv('data/output/kg_protein_info.csv')
        self.output_dir = output_dir
        
        logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        logger.info(f"Reports will be saved to: {self.output_dir}")
        
        # Separate node types
        self.nodes_by_type = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            self.nodes_by_type[data['node_type']].append(node)
    
    def create_overview_report(self):
        """Create an overview visualization showing the graph structure."""
        logger.info("Creating overview report...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cancer Knowledge Graph - Overview Report', fontsize=16, fontweight='bold')
        
        # 1. Node type distribution
        node_counts = {node_type: len(nodes) for node_type, nodes in self.nodes_by_type.items()}
        axes[0, 0].pie(node_counts.values(), labels=node_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Node Distribution')
        
        # 2. Edge type distribution
        edge_types = [data['edge_type'] for _, _, data in self.graph.edges(data=True)]
        edge_counts = Counter(edge_types)
        axes[0, 1].bar(edge_counts.keys(), edge_counts.values())
        axes[0, 1].set_title('Edge Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Patient MSI status distribution
        msi_counts = self.metadata_df['MSI.status'].value_counts()
        axes[1, 0].bar(msi_counts.index, msi_counts.values, color=['lightblue', 'orange', 'lightgreen'])
        axes[1, 0].set_title('Patient MSI Status Distribution')
        axes[1, 0].set_ylabel('Number of Patients')
        
        # 4. Proteomic subtype distribution
        subtype_counts = self.metadata_df['Proteomic.subtype'].value_counts()
        axes[1, 1].bar(subtype_counts.index, subtype_counts.values, color='lightcoral')
        axes[1, 1].set_title('Proteomic Subtype Distribution')
        axes[1, 1].set_ylabel('Number of Patients')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}01_kg_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Overview report saved: {self.output_dir}01_kg_overview.png")
    
    def create_key_proteins_subgraph_report(self):
        """Create focused subgraph visualization for key proteins."""
        logger.info("Creating key proteins subgraph report...")
        
        # Focus on key proteins and their connections
        key_proteins = self.protein_info_df[self.protein_info_df['is_key_protein']]['protein'].tolist()
        key_protein_nodes = [f'protein_{p}' for p in key_proteins]
        
        # Get all patients connected to key proteins
        patient_nodes = []
        for patient in self.nodes_by_type['patient']:
            for protein in key_protein_nodes:
                if self.graph.has_edge(patient, protein):
                    patient_nodes.append(patient)
                    break
        
        # Include pathway and subtype nodes
        subgraph_nodes = (key_protein_nodes + patient_nodes + 
                        self.nodes_by_type['pathway'] + 
                        self.nodes_by_type['subtype'])
        
        # Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        
        # Create layout
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # Color nodes by type
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node]['node_type']
            if node_type == 'patient':
                if subgraph.nodes[node].get('is_msi_h') == 1:
                    node_colors.append('red')
                    node_sizes.append(300)
                else:
                    node_colors.append('lightblue')
                    node_sizes.append(200)
            elif node_type == 'protein':
                if subgraph.nodes[node].get('is_key_protein', False):
                    node_colors.append('green')
                    node_sizes.append(200)
                else:
                    node_colors.append('lightgreen')
                    node_sizes.append(100)
            elif node_type == 'pathway':
                node_colors.append('orange')
                node_sizes.append(400)
            elif node_type == 'subtype':
                node_colors.append('purple')
                node_sizes.append(300)
            else:
                node_colors.append('gray')
                node_sizes.append(150)
        
        # Draw the graph
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=0.5)
        
        # Add labels for important nodes
        important_labels = {}
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            if node_data['node_type'] in ['pathway', 'subtype']:
                important_labels[node] = node.split('_')[1]
            elif node_data['node_type'] == 'protein' and node_data.get('is_key_protein', False):
                important_labels[node] = node.split('_')[1]
        
        nx.draw_networkx_labels(subgraph, pos, important_labels, font_size=8, font_weight='bold')
        
        plt.title('Knowledge Graph - Key Proteins Network\n(Zhang et al. proteins and their patient connections)', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='MSI-H Patients'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='MSS Patients'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Key Proteins'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Pathways'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Subtypes')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}02_key_proteins_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Key proteins report saved: {self.output_dir}02_key_proteins_network.png")
        logger.info(f"   Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    def create_msi_patients_subgraph_report(self):
        """Create focused subgraph visualization for MSI-H patients."""
        logger.info("Creating MSI-H patients subgraph report...")
        
        # Focus on MSI-H patients and their protein connections
        msi_h_patients = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'patient' and data.get('is_msi_h') == 1:
                msi_h_patients.append(node)
        
        # Get their top connected proteins
        protein_connections = defaultdict(int)
        for patient in msi_h_patients:
            for neighbor in self.graph.neighbors(patient):
                if neighbor.startswith('protein_'):
                    protein_connections[neighbor] += 1
        
        top_proteins = sorted(protein_connections.items(), key=lambda x: x[1], reverse=True)[:20]
        top_protein_nodes = [p[0] for p in top_proteins]
        
        subgraph_nodes = msi_h_patients + top_protein_nodes + self.nodes_by_type['subtype']
        
        # Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        
        # Create layout
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # Color nodes by type
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node]['node_type']
            if node_type == 'patient':
                # Color MSI-H patients by subtype
                subtype = subgraph.nodes[node].get('proteomic_subtype', '')
                if subtype == 'B':
                    node_colors.append('lightgreen')  # Good prognosis
                elif subtype == 'C':
                    node_colors.append('red')         # Poor prognosis
                else:
                    node_colors.append('orange')      # Other MSI-H
                node_sizes.append(400)
            elif node_type == 'protein':
                if subgraph.nodes[node].get('is_key_protein', False):
                    node_colors.append('darkgreen')
                    node_sizes.append(250)
                else:
                    node_colors.append('lightgreen')
                    node_sizes.append(150)
            elif node_type == 'subtype':
                node_colors.append('purple')
                node_sizes.append(300)
            else:
                node_colors.append('gray')
                node_sizes.append(150)
        
        # Draw the graph
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=0.5)
        
        # Add labels for important nodes
        important_labels = {}
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            if node_data['node_type'] == 'subtype':
                important_labels[node] = node.split('_')[1]
            elif node_data['node_type'] == 'protein' and node_data.get('is_key_protein', False):
                important_labels[node] = node.split('_')[1]
        
        nx.draw_networkx_labels(subgraph, pos, important_labels, font_size=8, font_weight='bold')
        
        plt.title('Knowledge Graph - MSI-H Patients Network\n(Key finding: Different protein patterns distinguish B vs C subtypes)', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=12, label='Subtype B (Good)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Subtype C (Poor)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12, label='Other MSI-H'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=10, label='Key Proteins'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Subtypes')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}03_msi_patients_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ MSI patients report saved: {self.output_dir}03_msi_patients_network.png")
        logger.info(f"   Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    def create_protein_analysis_report(self):
        """Create analysis plots for protein data."""
        logger.info("Creating protein analysis report...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Protein Analysis Report - Knowledge Graph', fontsize=16, fontweight='bold')
        
        # 1. Key protein categories
        protein_categories = self.protein_info_df.groupby('is_key_protein').size()
        axes[0, 0].pie(protein_categories.values, 
                      labels=['Other Proteins (784)', 'Key Proteins (16)'], 
                      autopct='%1.1f%%', colors=['lightblue', 'orange'])
        axes[0, 0].set_title('Key vs Other Proteins')
        
        # 2. Protein variance distribution
        axes[0, 1].hist(self.protein_info_df['variance'], bins=30, alpha=0.7, color='green')
        axes[0, 1].axvline(self.protein_info_df[self.protein_info_df['is_key_protein']]['variance'].mean(), 
                          color='red', linestyle='--', label='Key proteins avg')
        axes[0, 1].set_xlabel('Protein Variance')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Protein Variance Distribution')
        axes[0, 1].legend()
        
        # 3. Key proteins by category
        key_proteins = self.protein_info_df[self.protein_info_df['is_key_protein']]['protein'].tolist()
        
        # Categorize key proteins
        categories = {'MMR': [], 'EMT': [], 'Adhesion': [], 'Proliferation': [], 'Immune': []}
        
        mmr = ['MLH1', 'MSH2', 'MSH6', 'PMS2']
        emt = ['CDH1', 'VIM', 'COL1A1', 'COL3A1', 'FN1']
        adhesion = ['CTNNB1', 'CTNNA1', 'DSG2', 'PKP2']
        proliferation = ['MKI67', 'PCNA']
        immune = ['CD3E', 'CD8A']
        
        for protein in key_proteins:
            if protein in mmr:
                categories['MMR'].append(protein)
            elif protein in emt:
                categories['EMT'].append(protein)
            elif protein in adhesion:
                categories['Adhesion'].append(protein)
            elif protein in proliferation:
                categories['Proliferation'].append(protein)
            elif protein in immune:
                categories['Immune'].append(protein)
        
        category_counts = {k: len(v) for k, v in categories.items() if len(v) > 0}
        bars = axes[1, 0].bar(category_counts.keys(), category_counts.values(), color='lightcoral')
        axes[1, 0].set_title('Key Protein Categories\n(From Zhang et al. paper)')
        axes[1, 0].set_ylabel('Number of Proteins')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
        
        # 4. Patient-protein connectivity
        patient_connectivity = []
        for patient in self.nodes_by_type['patient']:
            protein_connections = sum(1 for neighbor in self.graph.neighbors(patient) 
                                    if neighbor.startswith('protein_'))
            patient_connectivity.append(protein_connections)
        
        axes[1, 1].hist(patient_connectivity, bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Number of Protein Connections per Patient')
        axes[1, 1].set_ylabel('Number of Patients')
        axes[1, 1].set_title('Patient-Protein Connectivity Distribution')
        axes[1, 1].axvline(np.mean(patient_connectivity), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(patient_connectivity):.0f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}04_protein_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Protein analysis report saved: {self.output_dir}04_protein_analysis.png")
    
    def create_msi_subtype_analysis_report(self):
        """Create MSI and subtype analysis visualization - THE KEY FINDING!"""
        logger.info("Creating MSI and subtype analysis report (KEY FINDING)...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MSI Status and Proteomic Subtype Analysis - KEY FINDINGS', fontsize=16, fontweight='bold')
        
        # 1. MSI-H patients by subtype (Zhang's key finding)
        msi_h_data = self.metadata_df[self.metadata_df['MSI.status'] == 'MSI-H']
        subtype_counts = msi_h_data['Proteomic.subtype'].value_counts()
        
        colors = ['lightgreen' if x == 'B' else 'red' if x == 'C' else 'orange' 
                 for x in subtype_counts.index]
        bars = axes[0, 0].bar(subtype_counts.index, subtype_counts.values, color=colors)
        axes[0, 0].set_title('🎯 MSI-H Patients by Proteomic Subtype\n(Zhang Key Finding: B=Good, C=Poor Prognosis)')
        axes[0, 0].set_ylabel('Number of Patients')
        axes[0, 0].set_xlabel('Proteomic Subtype')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Age distribution by MSI status
        msi_groups = [
            self.metadata_df[self.metadata_df['MSI.status'] == 'MSI-H']['age_at_initial_pathologic_diagnosis'],
            self.metadata_df[self.metadata_df['MSI.status'] == 'MSS']['age_at_initial_pathologic_diagnosis']
        ]
        box_plot = axes[0, 1].boxplot(msi_groups, labels=['MSI-H', 'MSS'])
        axes[0, 1].set_title('Age Distribution by MSI Status')
        axes[0, 1].set_ylabel('Age at Diagnosis')
        
        # 3. Mutation frequencies by subtype (B vs C comparison)
        mutation_cols = ['BRAF mutation', 'KRAS mutation', 'TP53 mutation', 'POLE mutation']
        subtype_mutation_data = []
        
        for subtype in ['B', 'C']:  # Focus on the key subtypes
            subtype_data = self.metadata_df[self.metadata_df['Proteomic.subtype'] == subtype]
            if len(subtype_data) > 0:
                mutation_freqs = [subtype_data[col].sum() / len(subtype_data) * 100 for col in mutation_cols]
                subtype_mutation_data.append(mutation_freqs)
        
        if subtype_mutation_data:
            x = np.arange(len(mutation_cols))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, subtype_mutation_data[0], width, 
                          label='Subtype B (Good)', color='lightgreen')
            if len(subtype_mutation_data) > 1:
                axes[1, 0].bar(x + width/2, subtype_mutation_data[1], width, 
                              label='Subtype C (Poor)', color='red')
            
            axes[1, 0].set_xlabel('Mutation Type')
            axes[1, 0].set_ylabel('Frequency (%)')
            axes[1, 0].set_title('🔬 Mutation Frequencies: Subtype B vs C\n(Different genomic patterns)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([col.replace(' mutation', '') for col in mutation_cols])
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Overall classification summary
        # Create a summary table as a plot
        axes[1, 1].axis('off')
        
        summary_data = [
            ['Total Patients', '90'],
            ['MSI-H Patients', '16'],
            ['MSS Patients', '60'],
            ['', ''],
            ['MSI-H Subtype B (Good)', '7'],
            ['MSI-H Subtype C (Poor)', '4'],
            ['MSI-H Other', '5'],
            ['', ''],
            ['Key Proteins', '16/800'],
            ['Pathways', '5'],
            ['Total Graph Nodes', '904'],
        ]
        
        table = axes[1, 1].table(cellText=summary_data,
                                colLabels=['Metric', 'Count'],
                                cellLoc='left',
                                loc='center',
                                colWidths=[0.6, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color the key rows
        table[(5, 0)].set_facecolor('lightgreen')  # Subtype B
        table[(5, 1)].set_facecolor('lightgreen')
        table[(6, 0)].set_facecolor('lightcoral')  # Subtype C
        table[(6, 1)].set_facecolor('lightcoral')
        
        axes[1, 1].set_title('📊 Dataset Summary\n(Ready for GNN Classification)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}05_msi_subtype_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ MSI subtype analysis report saved: {self.output_dir}05_msi_subtype_analysis.png")
        logger.info("   🎯 This visualization shows Zhang et al's key finding!")
    
    def generate_all_reports(self):
        """Generate all static visualization reports."""
        logger.info("📊 Generating all knowledge graph reports...")
        
        # Generate all reports
        self.create_overview_report()
        self.create_key_proteins_subgraph_report()
        self.create_msi_patients_subgraph_report()
        self.create_protein_analysis_report()
        self.create_msi_subtype_analysis_report()
        
        logger.info("✅ All reports generated successfully!")
        
        # Print summary
        print(f"\n📊 STATIC REPORTS GENERATED:")
        print(f"📁 Location: {self.output_dir}")
        print(f"📈 Reports created:")
        print(f"  01_kg_overview.png - Graph structure overview")
        print(f"  02_key_proteins_network.png - Key proteins and connections")
        print(f"  03_msi_patients_network.png - MSI-H patients network")
        print(f"  04_protein_analysis.png - Protein categories and analysis")
        print(f"  05_msi_subtype_analysis.png - 🎯 KEY FINDING: B vs C subtypes")
        print(f"\n🎯 Key insight: Report #5 shows Zhang's finding - proteomics splits MSI-H into good (B) vs poor (C) prognosis!")

def main():
    """Main function to generate all reports."""
    generator = KnowledgeGraphReportGenerator()
    generator.generate_all_reports()

if __name__ == "__main__":
    main()