import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class KnowledgeGraphVisualizer:
    """
    Visualize the cancer knowledge graph with multiple views and interactive plots.
    """
    
    def __init__(self, graph_path='data/output/cancer_knowledge_graph.pkl'):
        """Load the knowledge graph."""
        logger.info("Loading knowledge graph...")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Load additional data
        self.metadata_df = pd.read_csv('data/output/kg_patient_metadata.csv')
        self.protein_info_df = pd.read_csv('data/output/kg_protein_info.csv')
        
        logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Separate node types
        self.nodes_by_type = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            self.nodes_by_type[data['node_type']].append(node)
    
    def create_overview_visualization(self):
        """Create an overview visualization showing the graph structure."""
        logger.info("Creating overview visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Knowledge Graph Overview', fontsize=16, fontweight='bold')
        
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
        plt.savefig('data/output/kg_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_subgraph_visualization(self, focus='key_proteins'):
        """Create focused subgraph visualizations."""
        logger.info(f"Creating {focus} subgraph visualization...")
        
        if focus == 'key_proteins':
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
            
        elif focus == 'msi_patients':
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
        
        plt.title(f'Knowledge Graph - {focus.replace("_", " ").title()}', 
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
        plt.savefig(f'data/output/kg_subgraph_{focus}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Subgraph ({focus}): {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    def create_interactive_visualization(self):
        """Create an interactive visualization using Plotly."""
        logger.info("Creating interactive visualization...")
        
        # Create a smaller subgraph for better interactivity
        key_proteins = self.protein_info_df[self.protein_info_df['is_key_protein']]['protein'].tolist()
        key_protein_nodes = [f'protein_{p}' for p in key_proteins]
        
        # Get MSI-H patients
        msi_h_patients = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'patient' and data.get('is_msi_h') == 1:
                msi_h_patients.append(node)
        
        # Create subgraph
        interactive_nodes = (key_protein_nodes + msi_h_patients + 
                           self.nodes_by_type['pathway'] + 
                           self.nodes_by_type['subtype'] + 
                           self.nodes_by_type['mutation'])
        
        subgraph = self.graph.subgraph(interactive_nodes).copy()
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
        
        # Prepare node traces
        node_traces = {}
        color_map = {
            'patient': 'red',
            'protein': 'green', 
            'pathway': 'orange',
            'subtype': 'purple',
            'mutation': 'blue'
        }
        
        for node_type in color_map:
            node_traces[node_type] = go.Scatter(
                x=[], y=[], mode='markers+text',
                marker=dict(size=[], color=color_map[node_type], opacity=0.8),
                text=[], textposition="middle center",
                hovertemplate='<b>%{text}</b><br>Type: ' + node_type + '<extra></extra>',
                name=node_type.title(),
                showlegend=True
            )
        
        # Add nodes to traces
        for node in subgraph.nodes():
            x, y = pos[node]
            node_data = subgraph.nodes[node]
            node_type = node_data['node_type']
            
            node_traces[node_type]['x'] += tuple([x])
            node_traces[node_type]['y'] += tuple([y])
            
            # Node size based on connections
            degree = subgraph.degree(node)
            size = max(10, min(30, degree * 2))
            node_traces[node_type]['marker']['size'] += tuple([size])
            
            # Node label
            if node_type == 'patient':
                label = f"P{len(node_traces[node_type]['text']) + 1}"
            else:
                label = node.split('_')[1]
            node_traces[node_type]['text'] += tuple([label])
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none', showlegend=False
        )
        
        # Create the figure
        fig = go.Figure(data=[edge_trace] + list(node_traces.values()),
                       layout=go.Layout(
                           title=dict(
                               text='Interactive Cancer Knowledge Graph<br><sub>Key proteins, MSI-H patients, pathways, and subtypes</sub>',
                               font=dict(size=16)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Red: MSI-H patients, Green: Key proteins, Orange: Pathways, Purple: Subtypes, Blue: Mutations",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        # Save and show
        fig.write_html('data/output/interactive_knowledge_graph.html')
        fig.show()
        
        logger.info(f"Interactive graph saved to data/output/interactive_knowledge_graph.html")
    
    def create_protein_analysis_plots(self):
        """Create analysis plots for protein data."""
        logger.info("Creating protein analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Protein Analysis in Knowledge Graph', fontsize=16, fontweight='bold')
        
        # 1. Key protein categories
        protein_categories = self.protein_info_df.groupby('is_key_protein').size()
        axes[0, 0].pie(protein_categories.values, 
                      labels=['Other Proteins', 'Key Proteins'], 
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
        
        # 3. Key proteins by category (if we can infer from names)
        key_proteins = self.protein_info_df[self.protein_info_df['is_key_protein']]['protein'].tolist()
        
        # Categorize key proteins
        categories = {'MMR': [], 'EMT': [], 'Adhesion': [], 'Proliferation': [], 'Immune': [], 'Other': []}
        
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
            else:
                categories['Other'].append(protein)
        
        category_counts = {k: len(v) for k, v in categories.items() if len(v) > 0}
        axes[1, 0].bar(category_counts.keys(), category_counts.values(), color='lightcoral')
        axes[1, 0].set_title('Key Protein Categories')
        axes[1, 0].set_ylabel('Number of Proteins')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Patient-protein connectivity
        patient_connectivity = []
        for patient in self.nodes_by_type['patient']:
            protein_connections = sum(1 for neighbor in self.graph.neighbors(patient) 
                                    if neighbor.startswith('protein_'))
            patient_connectivity.append(protein_connections)
        
        axes[1, 1].hist(patient_connectivity, bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Number of Protein Connections')
        axes[1, 1].set_ylabel('Number of Patients')
        axes[1, 1].set_title('Patient-Protein Connectivity Distribution')
        
        plt.tight_layout()
        plt.savefig('data/output/protein_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_msi_subtype_analysis(self):
        """Create MSI and subtype analysis visualization."""
        logger.info("Creating MSI and subtype analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MSI Status and Proteomic Subtype Analysis', fontsize=16, fontweight='bold')
        
        # 1. MSI-H patients by subtype (Zhang's key finding)
        msi_h_data = self.metadata_df[self.metadata_df['MSI.status'] == 'MSI-H']
        subtype_counts = msi_h_data['Proteomic.subtype'].value_counts()
        
        colors = ['lightgreen' if x == 'B' else 'lightcoral' if x == 'C' else 'lightblue' 
                 for x in subtype_counts.index]
        bars = axes[0, 0].bar(subtype_counts.index, subtype_counts.values, color=colors)
        axes[0, 0].set_title('MSI-H Patients by Proteomic Subtype\n(Key Finding: B=Good, C=Poor Prognosis)')
        axes[0, 0].set_ylabel('Number of Patients')
        axes[0, 0].set_xlabel('Proteomic Subtype')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
        
        # 2. Age distribution by MSI status
        msi_groups = [
            self.metadata_df[self.metadata_df['MSI.status'] == 'MSI-H']['age_at_initial_pathologic_diagnosis'],
            self.metadata_df[self.metadata_df['MSI.status'] == 'MSS']['age_at_initial_pathologic_diagnosis']
        ]
        axes[0, 1].boxplot(msi_groups, labels=['MSI-H', 'MSS'])
        axes[0, 1].set_title('Age Distribution by MSI Status')
        axes[0, 1].set_ylabel('Age at Diagnosis')
        
        # 3. Mutation frequencies by subtype
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
            
            axes[1, 0].bar(x - width/2, subtype_mutation_data[0], width, label='Subtype B (Good)', color='lightgreen')
            if len(subtype_mutation_data) > 1:
                axes[1, 0].bar(x + width/2, subtype_mutation_data[1], width, label='Subtype C (Poor)', color='lightcoral')
            
            axes[1, 0].set_xlabel('Mutation Type')
            axes[1, 0].set_ylabel('Frequency (%)')
            axes[1, 0].set_title('Mutation Frequencies: Subtype B vs C')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([col.replace(' mutation', '') for col in mutation_cols])
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Tumor stage distribution
        stage_counts = self.metadata_df['stage_numeric'].value_counts().sort_index()
        axes[1, 1].bar(stage_counts.index, stage_counts.values, color='lightsalmon')
        axes[1, 1].set_title('Tumor Stage Distribution')
        axes[1, 1].set_xlabel('Tumor Stage')
        axes[1, 1].set_ylabel('Number of Patients')
        
        plt.tight_layout()
        plt.savefig('data/output/msi_subtype_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_visualizations(self):
        """Generate all visualization types."""
        logger.info("🎨 Generating all knowledge graph visualizations...")
        
        # 1. Overview
        self.create_overview_visualization()
        
        # 2. Subgraph visualizations
        self.create_subgraph_visualization('key_proteins')
        self.create_subgraph_visualization('msi_patients')
        
        # 3. Protein analysis
        self.create_protein_analysis_plots()
        
        # 4. MSI/subtype analysis
        self.create_msi_subtype_analysis()
        
        # 5. Interactive visualization
        self.create_interactive_visualization()
        
        logger.info("✅ All visualizations complete! Check data/output/ for files.")

def main():
    """Main function to create all visualizations."""
    visualizer = KnowledgeGraphVisualizer()
    visualizer.generate_all_visualizations()
    
    print("\n🎨 VISUALIZATION SUMMARY:")
    print("📊 Static plots saved as PNG files:")
    print("  - kg_overview.png")
    print("  - kg_subgraph_key_proteins.png") 
    print("  - kg_subgraph_msi_patients.png")
    print("  - protein_analysis.png")
    print("  - msi_subtype_analysis.png")
    print("\n🌐 Interactive plot:")
    print("  - interactive_knowledge_graph.html (open in browser)")

if __name__ == "__main__":
    main()