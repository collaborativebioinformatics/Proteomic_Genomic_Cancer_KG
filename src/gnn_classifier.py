import pickle
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class CancerGNN(nn.Module):
    """
    Graph Neural Network for cancer subtype classification.
    
    Uses Graph Attention Network (GAT) to integrate proteomic and genomic features
    for MSI vs MSS classification, replicating Zhang et al's key finding.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super(CancerGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph attention layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # Binary classification: MSI vs MSS
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph attention layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph-level representation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        x = self.classifier(x)
        return x

class KnowledgeGraphClassifier:
    """
    Complete pipeline for MSI vs MSS classification using Knowledge Graph + GNN.
    """
    
    def __init__(self, graph_path='data/output/cancer_knowledge_graph.pkl'):
        """Initialize the classifier with the knowledge graph."""
        logger.info("Loading knowledge graph for GNN classification...")
        
        # Load the graph and data
        with open(graph_path, 'rb') as f:
            self.nx_graph = pickle.load(f)
        
        self.metadata_df = pd.read_csv('data/output/kg_patient_metadata.csv')
        self.protein_df = pd.read_csv('data/output/kg_protein_abundance.csv', index_col=0)
        
        logger.info(f"Loaded graph: {self.nx_graph.number_of_nodes()} nodes, {self.nx_graph.number_of_edges()} edges")
        logger.info(f"Patient data: {len(self.metadata_df)} patients, {len(self.protein_df)} proteins")
        
        # Prepare data structures
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.node_features = None
        self.labels = None
        self.patient_indices = None
        self.pyg_data = None
    
    def prepare_graph_data(self):
        """Convert NetworkX graph to PyTorch Geometric format with node features."""
        logger.info("Preparing graph data for GNN...")
        
        # Get all nodes and create mapping
        all_nodes = list(self.nx_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Prepare node features
        node_features = []
        patient_indices = []
        labels = []
        
        for idx, node in enumerate(all_nodes):
            node_data = self.nx_graph.nodes[node]
            node_type = node_data['node_type']
            
            if node_type == 'patient':
                # Patient node features (clinical + genomic + proteomic)
                patient_id = node_data['tcga_id']
                
                # Clinical features
                patient_row = self.metadata_df[self.metadata_df['TCGA participant ID'] == patient_id].iloc[0]
                clinical_features = [
                    patient_row['age_at_initial_pathologic_diagnosis'] / 100.0,  # Normalize age
                    patient_row['stage_numeric'] / 4.0 if pd.notna(patient_row['stage_numeric']) else 0.5,
                    patient_row['gender_encoded'],
                    patient_row['site_right'],
                    patient_row['BRAF mutation'],
                    patient_row['KRAS mutation'],
                    patient_row['TP53 mutation'],
                    patient_row['POLE mutation'],
                ]
                
                # Proteomic features (top 10 most variable proteins for this patient)
                if patient_id in self.protein_df.columns:
                    protein_values = self.protein_df[patient_id].values
                    # Take top 10 most variable proteins to keep dimensionality manageable
                    top_protein_indices = np.argsort(self.protein_df.var(axis=1).values)[-10:]
                    proteomic_features = protein_values[top_protein_indices].tolist()
                else:
                    proteomic_features = [0.0] * 10  # Default if no protein data
                
                # Combine all features
                features = clinical_features + proteomic_features
                node_features.append(features)
                
                # Store patient info
                patient_indices.append(idx)
                # MSI = MSI-H OR MSI-L, MSS = MSS only
                msi_label = 1 if patient_row['MSI.status'] in ['MSI-H', 'MSI-L'] else 0
                labels.append(msi_label)
                
            elif node_type == 'protein':
                # Protein node features
                protein_symbol = node_data['gene_symbol']
                is_key = float(node_data['is_key_protein'])
                variance = node_data['variance']
                category_features = self._encode_protein_category(node_data['protein_category'])
                
                # Average expression across all patients
                if protein_symbol in self.protein_df.index:
                    avg_expression = self.protein_df.loc[protein_symbol].mean()
                else:
                    avg_expression = 0.0
                
                features = [is_key, variance, avg_expression] + category_features + [0.0] * 9  # Pad to match 18 total
                node_features.append(features)
                
            elif node_type == 'pathway':
                # Pathway node features
                pathway_name = node_data['pathway_name']
                pathway_features = self._encode_pathway_features(pathway_name)
                features = pathway_features + [0.0] * 10  # Pad to match 18 total
                node_features.append(features)
                
            else:
                # Other nodes (mutation, subtype) - simple encoding
                features = [1.0] + [0.0] * 17  # 18 features total
                node_features.append(features)
        
        # Convert to tensors
        self.node_features = torch.FloatTensor(node_features)
        self.labels = torch.LongTensor(labels)
        self.patient_indices = patient_indices
        
        # Create edge list
        edge_list = []
        for edge in self.nx_graph.edges():
            src_idx = node_to_idx[edge[0]]
            dst_idx = node_to_idx[edge[1]]
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])  # Make undirected
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        
        # Create PyTorch Geometric data
        self.pyg_data = Data(
            x=self.node_features,
            edge_index=edge_index,
            y=self.labels
        )
        
        logger.info(f"Graph data prepared:")
        logger.info(f"  Node features: {self.node_features.shape}")
        logger.info(f"  Edges: {edge_index.shape}")
        logger.info(f"  Patient nodes: {len(patient_indices)}")
        logger.info(f"  MSI patients (H+L): {self.labels.sum().item()}")
        logger.info(f"  MSS patients: {(self.labels == 0).sum().item()}")
    
    def _encode_protein_category(self, category):
        """One-hot encode protein categories."""
        categories = ['MMR', 'EMT', 'cell_adhesion', 'proliferation', 'immune', 'other']
        encoding = [0.0] * len(categories)
        if category in categories:
            encoding[categories.index(category)] = 1.0
        else:
            encoding[-1] = 1.0  # 'other'
        return encoding
    
    def _encode_pathway_features(self, pathway_name):
        """Encode pathway features."""
        pathways = ['DNA_Mismatch_Repair', 'Cell_Adhesion', 'ECM_Organization', 'Cell_Proliferation', 'Immune_Response']
        encoding = [0.0] * 8  # Leave space for other features
        if pathway_name in pathways:
            encoding[pathways.index(pathway_name)] = 1.0
        return encoding
    
    def create_subgraphs_for_patients(self):
        """Create individual subgraphs for each patient for training."""
        logger.info("Creating patient-centered subgraphs...")
        
        subgraphs = []
        for i, patient_idx in enumerate(self.patient_indices):
            # Get patient node and its neighbors (k-hop neighborhood)
            patient_node = list(self.nx_graph.nodes())[patient_idx]
            
            # Get 2-hop neighborhood
            neighbors = set([patient_node])
            for hop in range(2):
                new_neighbors = set()
                for node in neighbors:
                    new_neighbors.update(self.nx_graph.neighbors(node))
                neighbors.update(new_neighbors)
            
            # Create subgraph
            subgraph_nodes = list(neighbors)
            subgraph = self.nx_graph.subgraph(subgraph_nodes).copy()
            
            # Map to indices
            subgraph_node_to_idx = {node: idx for idx, node in enumerate(subgraph_nodes)}
            
            # Get features for subgraph nodes
            subgraph_features = []
            for node in subgraph_nodes:
                original_idx = list(self.nx_graph.nodes()).index(node)
                subgraph_features.append(self.node_features[original_idx])
            
            # Create edge list
            edge_list = []
            for edge in subgraph.edges():
                src_idx = subgraph_node_to_idx[edge[0]]
                dst_idx = subgraph_node_to_idx[edge[1]]
                edge_list.append([src_idx, dst_idx])
                edge_list.append([dst_idx, src_idx])
            
            if len(edge_list) > 0:
                edge_index = torch.LongTensor(edge_list).t().contiguous()
            else:
                edge_index = torch.LongTensor([[], []])
            
            # Create PyG data for this subgraph
            data = Data(
                x=torch.stack(subgraph_features),
                edge_index=edge_index,
                y=self.labels[i]
            )
            
            subgraphs.append(data)
        
        logger.info(f"Created {len(subgraphs)} patient subgraphs")
        return subgraphs
    
    def train_gnn(self, test_size=0.25, epochs=50, lr=0.01):
        """Train the Graph Neural Network."""
        logger.info("Training GNN classifier...")
        
        # Prepare data
        subgraphs = self.create_subgraphs_for_patients()
        
        # Split data
        X_indices = list(range(len(subgraphs)))
        y = [data.y.item() for data in subgraphs]
        
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            X_indices, y, test_size=test_size, random_state=42, stratify=y
        )
        
        train_data = [subgraphs[i] for i in X_train_idx]
        test_data = [subgraphs[i] for i in X_test_idx]
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
        
        # Initialize model
        input_dim = self.node_features.shape[1]
        model = CancerGNN(input_dim=input_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        predictions = []
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch)
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())  # MSI-H probability
                true_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        auc_score = roc_auc_score(true_labels, probabilities)
        
        results = {
            'model': model,
            'train_losses': train_losses,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_auc': auc_score,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'y_train': y_train,
            'y_test': y_test
        }
        
        logger.info(f"GNN Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  AUC: {auc_score:.4f}")
        
        return results
    
    def train_baseline_models(self):
        """Train baseline models for comparison."""
        logger.info("Training baseline models...")
        
        # Prepare tabular data for baseline models
        patient_features = []
        labels = []
        
        for patient_idx in self.patient_indices:
            # Get patient data
            node = list(self.nx_graph.nodes())[patient_idx]
            node_data = self.nx_graph.nodes[node]
            patient_id = node_data['tcga_id']
            
            # Get clinical + genomic features
            patient_row = self.metadata_df[self.metadata_df['TCGA participant ID'] == patient_id].iloc[0]
            
            clinical_features = [
                patient_row['age_at_initial_pathologic_diagnosis'],
                patient_row['stage_numeric'] if pd.notna(patient_row['stage_numeric']) else 2.0,
                patient_row['gender_encoded'],
                patient_row['site_right'],
                patient_row['BRAF mutation'],
                patient_row['KRAS mutation'],
                patient_row['TP53 mutation'],
                patient_row['POLE mutation'],
            ]
            
            # Get proteomic features (all available proteins)
            if patient_id in self.protein_df.columns:
                proteomic_features = self.protein_df[patient_id].values.tolist()
            else:
                proteomic_features = [0.0] * len(self.protein_df)
            
            # Combine features
            features = clinical_features + proteomic_features
            patient_features.append(features)
            
            # MSI = MSI-H OR MSI-L, MSS = MSS only
            msi_label = 1 if patient_row['MSI.status'] in ['MSI-H', 'MSI-L'] else 0
            labels.append(msi_label)
        
        # Convert to numpy arrays
        X = np.array(patient_features)
        y = np.array(labels)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        baseline_results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            
            # Train on full data for detailed evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.25, random_state=42, stratify=y
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
            auc_score = roc_auc_score(y_test, probabilities)
            
            baseline_results[name] = {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'test_auc': auc_score,
                'predictions': predictions,
                'probabilities': probabilities,
                'true_labels': y_test
            }
            
            logger.info(f"{name} Results:")
            logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            logger.info(f"  Test Accuracy: {accuracy:.4f}")
            logger.info(f"  AUC: {auc_score:.4f}")
        
        return baseline_results
    
    def create_evaluation_report(self, gnn_results, baseline_results, output_dir='reports/'):
        """Create comprehensive evaluation report with visualizations."""
        logger.info("Creating evaluation report...")
        
        # Prepare results summary
        results_summary = []
        
        # GNN results
        results_summary.append({
            'Model': 'Graph Neural Network',
            'Accuracy': gnn_results['test_accuracy'],
            'Precision': gnn_results['test_precision'],
            'Recall': gnn_results['test_recall'],
            'F1-Score': gnn_results['test_f1'],
            'AUC': gnn_results['test_auc']
        })
        
        # Baseline results
        for name, results in baseline_results.items():
            results_summary.append({
                'Model': name,
                'Accuracy': results['test_accuracy'],
                'Precision': results['test_precision'],
                'Recall': results['test_recall'],
                'F1-Score': results['test_f1'],
                'AUC': results['test_auc']
            })
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MSI vs MSS Classification Results - Knowledge Graph GNN', 
                     fontsize=16, fontweight='bold')
        
        # 1. Model comparison
        df_results = pd.DataFrame(results_summary)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(df_results['Model']):
            values = [df_results.iloc[i][metric] for metric in metrics]
            color = 'red' if model == 'Graph Neural Network' else 'lightblue'
            axes[0, 0].bar(x + i*width, values, width, label=model, color=color, alpha=0.8)
        
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('🎯 Model Performance Comparison')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(metrics, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. GNN Training curve
        axes[0, 1].plot(gnn_results['train_losses'], color='red', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Training Loss')
        axes[0, 1].set_title('GNN Training Progress')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. ROC Curves
        from sklearn.metrics import roc_curve
        
        # GNN ROC
        fpr_gnn, tpr_gnn, _ = roc_curve(gnn_results['true_labels'], gnn_results['probabilities'])
        axes[1, 0].plot(fpr_gnn, tpr_gnn, color='red', linewidth=2, 
                       label=f'GNN (AUC = {gnn_results["test_auc"]:.3f})')
        
        # Baseline ROCs
        colors = ['blue', 'green', 'orange']
        for i, (name, results) in enumerate(baseline_results.items()):
            fpr, tpr, _ = roc_curve(results['true_labels'], results['probabilities'])
            axes[1, 0].plot(fpr, tpr, color=colors[i], linewidth=2, 
                           label=f'{name} (AUC = {results["test_auc"]:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves - MSI vs MSS Classification')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix for GNN
        cm = confusion_matrix(gnn_results['true_labels'], gnn_results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   xticklabels=['MSS', 'MSI'], yticklabels=['MSS', 'MSI'])
        axes[1, 1].set_title('GNN Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}06_gnn_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create results table
        print("\n" + "="*80)
        print("🎯 MSI vs MSS CLASSIFICATION RESULTS")
        print("="*80)
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
        print("-"*80)
        
        for _, row in df_results.iterrows():
            print(f"{row['Model']:<25} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} "
                  f"{row['Recall']:<10.4f} {row['F1-Score']:<10.4f} {row['AUC']:<10.4f}")
        
        print("="*80)
        print("🎯 KEY FINDINGS:")
        
        gnn_acc = gnn_results['test_accuracy']
        best_baseline_acc = max([r['test_accuracy'] for r in baseline_results.values()])
        
        if gnn_acc > best_baseline_acc:
            improvement = ((gnn_acc - best_baseline_acc) / best_baseline_acc) * 100
            print(f"✅ GNN outperforms baselines by {improvement:.1f}% accuracy!")
            print("✅ Knowledge graph structure provides valuable information for classification")
        else:
            print("📊 GNN performance comparable to baselines")
            print("💡 Graph structure may need refinement or more sophisticated GNN architecture")
        
        print(f"📈 Dataset: {len(gnn_results['true_labels'])} test patients")
        print(f"📈 MSI patients: {sum(gnn_results['true_labels'])} ({sum(gnn_results['true_labels'])/len(gnn_results['true_labels'])*100:.1f}%)")
        print(f"📈 MSS patients: {len(gnn_results['true_labels']) - sum(gnn_results['true_labels'])} ({(len(gnn_results['true_labels']) - sum(gnn_results['true_labels']))/len(gnn_results['true_labels'])*100:.1f}%)")
        
        logger.info(f"✅ Evaluation report saved: {output_dir}06_gnn_evaluation_results.png")
        
        return df_results
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline."""
        logger.info("🚀 Starting complete MSI vs MSS classification evaluation...")
        
        # Prepare data
        self.prepare_graph_data()
        
        # Train models
        logger.info("📊 Training Graph Neural Network...")
        gnn_results = self.train_gnn()
        
        logger.info("📊 Training baseline models...")
        baseline_results = self.train_baseline_models()
        
        # Create evaluation report
        logger.info("📊 Creating evaluation report...")
        results_df = self.create_evaluation_report(gnn_results, baseline_results)
        
        logger.info("✅ Complete evaluation finished!")
        
        return {
            'gnn_results': gnn_results,
            'baseline_results': baseline_results,
            'summary': results_df
        }

def main():
    """Main function to run GNN classification."""
    classifier = KnowledgeGraphClassifier()
    results = classifier.run_complete_evaluation()
    return results

if __name__ == "__main__":
    main()