import spacy
import sys
import os
from typing import List, Dict

# Add src directory to path
sys.path.append(os.path.dirname(__file__))
from exporter import export_analysis_to_csv

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def analyze_sentence(text: str) -> List[Dict]:
    """
    Analyze a single sentence and return token data.
    
    Args:
        text (str): Input sentence
        
    Returns:
        List[Dict]: List of token dictionaries
    """
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        token_data = {
            'sentence': text,
            'token': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'dependency': token.dep_
        }
        tokens.append(token_data)
    
    return tokens


def process_file(input_txt_path: str, output_csv_path: str):
    """
    Process a text file line by line and export analysis to CSV.
    
    Args:
        input_txt_path (str): Path to input text file
        output_csv_path (str): Path to output CSV file
    """
    all_results = []
    
    try:
        with open(input_txt_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Analyze the sentence
                try:
                    tokens = analyze_sentence(line)
                    all_results.extend(tokens)
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue
        
        # Export all results to CSV
        export_analysis_to_csv(all_results, output_csv_path)
        print(f"Successfully processed {len(all_results)} tokens from {input_txt_path}")
        print(f"Results exported to {output_csv_path}")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_txt_path}")
    except Exception as e:
        raise Exception(f"Error processing file {input_txt_path}: {str(e)}")


def process_datasets_directory(datasets_dir: str = "datasets"):
    """
    Process all .txt files in the datasets directory.
    
    Args:
        datasets_dir (str): Path to datasets directory
    """
    import glob
    
    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Find all .txt files in datasets directory
    txt_files = glob.glob(os.path.join(datasets_dir, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {datasets_dir}")
        return
    
    print(f"Found {len(txt_files)} .txt files in {datasets_dir}")
    
    for txt_file in txt_files:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        output_csv = os.path.join(outputs_dir, f"{base_name}_analysis.csv")
        
        print(f"\nProcessing {txt_file}...")
        try:
            process_file(txt_file, output_csv)
        except Exception as e:
            print(f"Failed to process {txt_file}: {str(e)}")


def process_single_file(input_path: str, output_path: str = None):
    """
    Process a single text file.
    
    Args:
        input_path (str): Path to input text file
        output_path (str): Path to output CSV file (optional)
    """
    if output_path is None:
        # Generate default output path
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        output_path = os.path.join(outputs_dir, f"{base_name}_analysis.csv")
    
    print(f"Processing {input_path}...")
    process_file(input_path, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process text files for morphological analysis")
    parser.add_argument("--input", "-i", help="Input text file path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--datasets", "-d", action="store_true", 
                       help="Process all .txt files in datasets/ directory")
    
    args = parser.parse_args()
    
    if args.datasets:
        process_datasets_directory()
    elif args.input:
        process_single_file(args.input, args.output)
    else:
        print("Usage:")
        print("  python batch_processor.py --input file.txt")
        print("  python batch_processor.py --input file.txt --output results.csv")
        print("  python batch_processor.py --datasets")
