#!/usr/bin/env python3
"""
Simple runner for GNN MSI-H vs MSS classification.
"""
import sys
from src.gnn_classifier import KnowledgeGraphClassifier

def run_gnn_quick_test():
    """Run a quick test with 1 epoch to verify everything works."""
    print("🚀 Running GNN Quick Test (1 epoch)...")
    
    classifier = KnowledgeGraphClassifier()
    # Override epochs for quick test
    classifier.prepare_graph_data()
    gnn_results = classifier.train_gnn(epochs=1)
    
    print(f"\n✅ Quick Test Results:")
    print(f"   Accuracy: {gnn_results['test_accuracy']:.4f}")
    print(f"   AUC: {gnn_results['test_auc']:.4f}")
    print(f"   Test patients: {len(gnn_results['true_labels'])}")
    print(f"   MSI patients: {sum(gnn_results['true_labels'])}")
    
    return gnn_results

def run_gnn_full_evaluation():
    """Run the complete evaluation with baselines."""
    print("🚀 Running Full GNN Evaluation...")
    
    classifier = KnowledgeGraphClassifier()
    results = classifier.run_complete_evaluation()
    
    return results

def main():
    """Main runner with options."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode in ['test', 'quick', 't']:
            run_gnn_quick_test()
        elif mode in ['full', 'complete', 'f']:
            run_gnn_full_evaluation()
        else:
            print("❌ Unknown mode. Use 'test' for quick test or 'full' for complete evaluation.")
            sys.exit(1)
    else:
        # Default: ask user
        print("🎯 MSI vs MSS Classification using Knowledge Graph + GNN")
        print("\nChoose mode:")
        print("  1. Quick test (1 epoch) - verify everything works")
        print("  2. Full evaluation (50 epochs + baselines) - complete analysis")
        
        choice = input("\nEnter choice (1/2): ").strip()
        
        if choice == '1':
            run_gnn_quick_test()
        elif choice == '2':
            run_gnn_full_evaluation()
        else:
            print("❌ Invalid choice. Exiting.")
            sys.exit(1)

if __name__ == "__main__":
    main()