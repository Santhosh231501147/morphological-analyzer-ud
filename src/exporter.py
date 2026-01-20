import csv


def export_analysis_to_csv(results, output_path):
    """
    Export morphological analysis results to CSV file.
    
    Args:
        results (list): List of dictionaries containing analysis results
        output_path (str): Path to save the CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sentence', 'token', 'lemma', 'pos', 'dependency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)


def export_user_input_analysis(text, analysis_results, output_path):
    """
    Export user input analysis to CSV.
    
    Args:
        text (str): Original input text
        analysis_results (list): List of token analysis results
        output_path (str): Path to save the CSV file
    """
    csv_results = []
    
    for token_data in analysis_results:
        csv_results.append({
            'sentence': text,
            'token': token_data.get('FORM', ''),
            'lemma': token_data.get('LEMMA', ''),
            'pos': token_data.get('UPOS', ''),
            'dependency': token_data.get('DEPREL', '')
        })
    
    export_analysis_to_csv(csv_results, output_path)


def export_ud_dataset_analysis(sentences, output_path):
    """
    Export UD dataset analysis to CSV.
    
    Args:
        sentences (list): List of sentences with token data
        output_path (str): Path to save the CSV file
    """
    csv_results = []
    
    for sentence_idx, sentence in enumerate(sentences):
        # Reconstruct sentence text
        sentence_text = " ".join([token['FORM'] for token in sentence])
        
        for token in sentence:
            csv_results.append({
                'sentence': sentence_text,
                'token': token.get('FORM', ''),
                'lemma': token.get('LEMMA', ''),
                'pos': token.get('UPOS', ''),
                'dependency': token.get('DEPREL', '')
            })
    
    export_analysis_to_csv(csv_results, output_path)
