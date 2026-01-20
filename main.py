import sys
import os

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ud_loader import load_conllu
from evaluation import evaluate_pos_accuracy, build_confusion_matrix, save_confusion_csv
from exporter import export_user_input_analysis, export_ud_dataset_analysis
from visualization import plot_accuracy_bar, plot_confusion_heatmap
from batch_processor import process_datasets_directory, process_single_file


def analyze_user_input():
    """Analyze morphological features of user input text."""
    print("Enter text to analyze:")
    text = input("> ")
    
    # Simple tokenization (split by whitespace)
    tokens = text.split()
    
    print("\nMorphological Analysis:")
    print("-" * 50)
    
    for i, token in enumerate(tokens, 1):
        print(f"{i}. {token}")
        print(f"   FORM: {token}")
        print(f"   LEMMA: {token.lower()}")  # Simple lemma
        print(f"   UPOS: NOUN")  # Default POS
        print(f"   HEAD: {i-1 if i > 1 else 0}")
        print(f"   DEPREL: root" if i == 1 else f"   DEPREL: dep")
        print()
    
    return text, tokens


def analyze_ud_dataset():
    """Analyze UD dataset sample."""
    dataset_path = "dataset/UD_English-EWT/en_ewt-ud-train.conllu"
    
    try:
        # Load the dataset
        sentences = load_conllu(dataset_path)
        
        # Display first 3 sentences
        print(f"\nFirst 3 sentences from {dataset_path}:")
        print("=" * 60)
        
        for sent_idx, sentence in enumerate(sentences[:3], 1):
            print(f"\nSentence {sent_idx}:")
            print("-" * 30)
            
            for token_idx, token in enumerate(sentence, 1):
                print(f"{token_idx}. {token['FORM']}")
                print(f"   FORM: {token['FORM']}")
                print(f"   LEMMA: {token['LEMMA']}")
                print(f"   UPOS: {token['UPOS']}")
                print(f"   HEAD: {token['HEAD']}")
                print(f"   DEPREL: {token['DEPREL']}")
                print()
            
            print("=" * 60)
            
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please ensure the UD_English-EWT dataset is properly downloaded.")
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
    
    return sentences


def evaluate_pos_accuracy_menu():
    """Evaluate POS accuracy on UD dataset."""
    dataset_path = "dataset/UD_English-EWT/en_ewt-ud-train.conllu"
    
    try:
        print("\nEvaluating POS accuracy...")
        results = evaluate_pos_accuracy(dataset_path)
        
        print(f"\nAccuracy Results:")
        print("-" * 30)
        print(f"Total tokens: {results['total']}")
        print(f"Correct predictions: {results['correct']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        
        # Generate plot
        plot_accuracy_bar(results['accuracy'])
        print(f"Accuracy plot saved to outputs/accuracy.png")
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please ensure the UD_English-EWT dataset is properly downloaded.")
    except Exception as e:
        print(f"Error evaluating accuracy: {str(e)}")


def generate_confusion_matrix_menu():
    """Generate confusion matrix for POS tagging."""
    dataset_path = "dataset/UD_English-EWT/en_ewt-ud-train.conllu"
    
    try:
        print("\nGenerating confusion matrix...")
        matrix = build_confusion_matrix(dataset_path)
        
        print(f"\nConfusion Matrix:")
        print("-" * 30)
        
        # Display matrix summary
        total_predictions = sum(sum(row.values()) for row in matrix.values())
        print(f"Total predictions: {total_predictions}")
        print(f"Unique POS tags: {len(matrix)}")
        
        # Save to CSV
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"outputs/confusion_matrix_{timestamp}.csv"
        save_confusion_csv(matrix, csv_path)
        print(f"Confusion matrix saved to {csv_path}")
        
        # Generate plot
        plot_confusion_heatmap(matrix)
        print(f"Confusion matrix heatmap saved to outputs/confusion_matrix.png")
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please ensure the UD_English-EWT dataset is properly downloaded.")
    except Exception as e:
        print(f"Error generating confusion matrix: {str(e)}")


def export_csv_menu():
    """Export analysis results to CSV."""
    print("\nExport Options:")
    print("1. Export user input analysis")
    print("2. Export UD dataset analysis")
    
    choice = input("Select export option (1-2): ").strip()
    
    if choice == '1':
        text, tokens = analyze_user_input()
        
        # Convert to expected format
        analysis_results = []
        for i, token in enumerate(tokens, 1):
            analysis_results.append({
                'FORM': token,
                'LEMMA': token.lower(),
                'UPOS': 'NOUN',
                'DEPREL': 'root' if i == 1 else 'dep'
            })
        
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/user_analysis_{timestamp}.csv"
        export_user_input_analysis(text, analysis_results, output_path)
        print(f"Analysis exported to {output_path}")
        
    elif choice == '2':
        sentences = analyze_ud_dataset()
        
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/ud_analysis_{timestamp}.csv"
        export_ud_dataset_analysis(sentences, output_path)
        print(f"UD dataset analysis exported to {output_path}")
        
    else:
        print("Invalid choice.")


def batch_processing_menu():
    """Batch processing menu."""
    print("\nBatch Processing Options:")
    print("1. Process single file")
    print("2. Process all files in datasets/ directory")
    
    choice = input("Select option (1-2): ").strip()
    
    if choice == '1':
        input_path = input("Enter input file path: ").strip()
        output_path = input("Enter output CSV path (optional): ").strip()
        
        if not output_path:
            output_path = None
        
        try:
            process_single_file(input_path, output_path)
        except Exception as e:
            print(f"Error: {str(e)}")
            
    elif choice == '2':
        try:
            process_datasets_directory()
        except Exception as e:
            print(f"Error: {str(e)}")
            
    else:
        print("Invalid choice.")


def start_api_server():
    """Start the REST API server."""
    print("\nStarting REST API server...")
    print("Server will be available at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        import uvicorn
        from api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")
    except Exception as e:
        print(f"Error starting API server: {str(e)}")


def start_gui():
    """Start the GUI interface."""
    print("\nStarting GUI interface...")
    print("GUI will be available in your default browser")
    print("Press Ctrl+C in terminal to stop the GUI")
    
    try:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/gui.py"])
    except Exception as e:
        print(f"Error starting GUI: {str(e)}")
        print("Make sure streamlit is installed: pip install streamlit")


def main():
    """Main menu for morphological analyzer."""
    while True:
        print("\nMorphological Analyzer")
        print("=" * 30)
        print("1. User input analysis")
        print("2. UD dataset analysis")
        print("3. POS accuracy evaluation")
        print("4. Generate confusion matrix")
        print("5. Export CSV")
        print("6. Batch processing")
        print("7. Start REST API")
        print("8. Start GUI")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == '1':
            analyze_user_input()
        elif choice == '2':
            analyze_ud_dataset()
        elif choice == '3':
            evaluate_pos_accuracy_menu()
        elif choice == '4':
            generate_confusion_matrix_menu()
        elif choice == '5':
            export_csv_menu()
        elif choice == '6':
            batch_processing_menu()
        elif choice == '7':
            start_api_server()
        elif choice == '8':
            start_gui()
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 0-8.")


if __name__ == "__main__":
    main()
