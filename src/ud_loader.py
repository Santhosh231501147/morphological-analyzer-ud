def load_conllu(file_path):
    """
    Read .conllu file and extract linguistic features.
    
    Args:
        file_path (str): Path to the .conllu file
        
    Returns:
        list: List of sentences where each sentence is a list of token dictionaries
              with keys: FORM, LEMMA, UPOS, HEAD, DEPREL
    """
    sentences = []
    current_sentence = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                
                # Skip empty lines (sentence boundaries)
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                # Skip comment lines starting with #
                if line.startswith('#'):
                    continue
                
                # Parse token line
                parts = line.split('\t')
                if len(parts) >= 10:
                    token_dict = {
                        'FORM': parts[1],
                        'LEMMA': parts[2],
                        'UPOS': parts[3],
                        'HEAD': parts[6],
                        'DEPREL': parts[7]
                    }
                    current_sentence.append(token_dict)
            
            # Add last sentence if file doesn't end with empty line
            if current_sentence:
                sentences.append(current_sentence)
                
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    return sentences


def get_sample_sentences(n=5):
    """
    Return first n sentences from train dataset.
    
    Args:
        n (int): Number of sentences to return (default: 5)
        
    Returns:
        list: List of n sentences from train dataset
    """
    try:
        return load_conllu('train.conllu')[:n]
    except FileNotFoundError:
        raise FileNotFoundError("train.conllu file not found. Make sure the training data file exists.")
    except Exception as e:
        raise Exception(f"Error loading sample sentences: {str(e)}")
