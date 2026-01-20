import sys
import os
import spacy

# Add src directory to path to import ud_loader
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ud_loader import load_conllu


def evaluate_pos_accuracy(conllu_path):
    """
    Evaluate POS tagging accuracy against gold standard.
    
    Args:
        conllu_path (str): Path to the .conllu file with gold annotations
        
    Returns:
        dict: Dictionary with total tokens, correct predictions, and accuracy
    """
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Load gold data
    gold_sentences = load_conllu(conllu_path)
    
    total_tokens = 0
    correct_tokens = 0
    
    for sentence in gold_sentences:
        # Reconstruct sentence text from FORM fields
        sentence_text = " ".join([token['FORM'] for token in sentence])
        
        # Run spaCy model
        doc = nlp(sentence_text)
        
        # Align tokens and compare POS tags
        min_length = min(len(sentence), len(doc))
        
        for i in range(min_length):
            gold_token = sentence[i]
            spacy_token = doc[i]
            
            total_tokens += 1
            
            # Compare spaCy POS with gold UPOS
            if spacy_token.pos_ == gold_token['UPOS']:
                correct_tokens += 1
        
        # Handle token length mismatches
        if len(sentence) != len(doc):
            # Count remaining tokens as incorrect
            remaining_tokens = abs(len(sentence) - len(doc))
            total_tokens += remaining_tokens
    
    # Calculate accuracy
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return {
        "total": total_tokens,
        "correct": correct_tokens,
        "accuracy": accuracy
    }


def build_confusion_matrix(conllu_path):
    """
    Build confusion matrix for POS tagging.
    
    Args:
        conllu_path (str): Path to the .conllu file with gold annotations
        
    Returns:
        dict: Confusion matrix dictionary [gold_pos][pred_pos] = count
    """
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Load gold data
    gold_sentences = load_conllu(conllu_path)
    
    # Initialize confusion matrix
    confusion_matrix = {}
    
    for sentence in gold_sentences:
        # Reconstruct sentence text from FORM fields
        sentence_text = " ".join([token['FORM'] for token in sentence])
        
        # Run spaCy model
        doc = nlp(sentence_text)
        
        # Align tokens and update confusion matrix
        min_length = min(len(sentence), len(doc))
        
        for i in range(min_length):
            gold_token = sentence[i]
            spacy_token = doc[i]
            
            gold_pos = gold_token['UPOS']
            pred_pos = spacy_token.pos_
            
            # Initialize nested dictionary if needed
            if gold_pos not in confusion_matrix:
                confusion_matrix[gold_pos] = {}
            
            # Increment count
            if pred_pos not in confusion_matrix[gold_pos]:
                confusion_matrix[gold_pos][pred_pos] = 0
            confusion_matrix[gold_pos][pred_pos] += 1
        
        # Handle token length mismatches
        if len(sentence) != len(doc):
            # Count remaining tokens as mismatches
            if len(sentence) > len(doc):
                # Gold tokens without predictions
                for i in range(len(doc), len(sentence)):
                    gold_pos = sentence[i]['UPOS']
                    pred_pos = "MISSING"
                    
                    if gold_pos not in confusion_matrix:
                        confusion_matrix[gold_pos] = {}
                    if pred_pos not in confusion_matrix[gold_pos]:
                        confusion_matrix[gold_pos][pred_pos] = 0
                    confusion_matrix[gold_pos][pred_pos] += 1
            else:
                # Predictions without gold tokens
                for i in range(len(sentence), len(doc)):
                    gold_pos = "MISSING"
                    pred_pos = doc[i].pos_
                    
                    if gold_pos not in confusion_matrix:
                        confusion_matrix[gold_pos] = {}
                    if pred_pos not in confusion_matrix[gold_pos]:
                        confusion_matrix[gold_pos][pred_pos] = 0
                    confusion_matrix[gold_pos][pred_pos] += 1
    
    return confusion_matrix


def save_confusion_csv(matrix, output_path):
    """
    Save confusion matrix to CSV file.
    
    Args:
        matrix (dict): Confusion matrix dictionary
        output_path (str): Path to save the CSV file
    """
    import csv
    
    # Get all unique POS tags (both gold and predicted)
    all_pos_tags = set()
    for gold_pos, predictions in matrix.items():
        all_pos_tags.add(gold_pos)
        for pred_pos in predictions:
            all_pos_tags.add(pred_pos)
    
    # Sort POS tags for consistent ordering
    pos_tags = sorted(all_pos_tags)
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['Gold\\Pred'] + pos_tags
        writer.writerow(header)
        
        # Write rows
        for gold_pos in pos_tags:
            row = [gold_pos]
            for pred_pos in pos_tags:
                count = matrix.get(gold_pos, {}).get(pred_pos, 0)
                row.append(count)
            writer.writerow(row)
