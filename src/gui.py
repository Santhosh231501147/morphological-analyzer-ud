import streamlit as st
import pandas as pd
import spacy
import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(__file__))
from ud_loader import load_conllu
from evaluation import evaluate_pos_accuracy, build_confusion_matrix, save_confusion_csv
from exporter import export_user_input_analysis, export_ud_dataset_analysis
from visualization import plot_accuracy_bar, plot_confusion_heatmap

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


def analyze_text(text):
    """Analyze text and return results."""
    doc = nlp(text)
    
    tokens_data = []
    noun_count = 0
    verb_count = 0
    noun_phrases = []
    verb_phrases = []
    
    for token in doc:
        token_data = {
            'Token': token.text,
            'Lemma': token.lemma_,
            'POS': token.pos_,
            'Dependency': token.dep_,
            'Head': token.head.text
        }
        tokens_data.append(token_data)
        
        # Count nouns and verbs
        if token.pos_ == 'NOUN':
            noun_count += 1
        elif token.pos_ == 'VERB':
            verb_count += 1
    
    # Extract noun and verb phrases
    for chunk in doc.noun_chunks:
        noun_phrases.append(chunk.text)
    
    # Simple verb phrase detection (auxiliary + verb)
    for i, token in enumerate(doc):
        if token.pos_ == 'VERB' and i > 0 and doc[i-1].pos_ == 'AUX':
            verb_phrases.append(f"{doc[i-1].text} {token.text}")
        elif token.pos_ == 'VERB' and (i == 0 or doc[i-1].pos_ != 'AUX'):
            verb_phrases.append(token.text)
    
    return tokens_data, noun_count, verb_count, noun_phrases, verb_phrases


def main():
    st.title("Morphological Analyzer")
    st.markdown("Analyze text morphology with spaCy and Universal Dependencies")
    
    # Text input section
    st.header("Text Analysis")
    
    text_input = st.text_area("Enter text to analyze:", height=100, 
                             placeholder="Type or paste your text here...")
    
    if st.button("Analyze Text") and text_input.strip():
        with st.spinner("Analyzing text..."):
            tokens_data, noun_count, verb_count, noun_phrases, verb_phrases = analyze_text(text_input)
            
            # Display tokens table
            st.subheader("Token Analysis")
            df = pd.DataFrame(tokens_data)
            st.dataframe(df, use_container_width=True)
            
            # Display statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nouns", noun_count)
            with col2:
                st.metric("Verbs", verb_count)
            
            # Display phrases
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Noun Phrases")
                st.write(noun_phrases if noun_phrases else ["None found"])
            
            with col4:
                st.subheader("Verb Phrases")
                st.write(verb_phrases if verb_phrases else ["None found"])
    
    st.divider()
    
    # Evaluation section
    st.header("Dataset Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Run Accuracy Evaluation"):
            with st.spinner("Evaluating accuracy..."):
                try:
                    dataset_path = "dataset/UD_English-EWT/en_ewt-ud-train.conllu"
                    results = evaluate_pos_accuracy(dataset_path)
                    
                    st.success(f"Accuracy: {results['accuracy']:.4f}")
                    st.write(f"Total tokens: {results['total']}")
                    st.write(f"Correct predictions: {results['correct']}")
                    
                    # Generate and display accuracy plot
                    plot_accuracy_bar(results['accuracy'])
                    st.image("outputs/accuracy.png")
                    
                except FileNotFoundError:
                    st.error("UD dataset not found. Please ensure the dataset is properly downloaded.")
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
    
    with col2:
        if st.button("Generate Confusion Matrix"):
            with st.spinner("Generating confusion matrix..."):
                try:
                    dataset_path = "dataset/UD_English-EWT/en_ewt-ud-train.conllu"
                    matrix = build_confusion_matrix(dataset_path)
                    
                    st.json(matrix)
                    
                    # Generate and display heatmap
                    plot_confusion_heatmap(matrix)
                    st.image("outputs/confusion_matrix.png")
                    
                except FileNotFoundError:
                    st.error("UD dataset not found. Please ensure the dataset is properly downloaded.")
                except Exception as e:
                    st.error(f"Confusion matrix generation failed: {str(e)}")
    
    st.divider()
    
    # Export section
    st.header("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Analysis to CSV") and text_input.strip():
            with st.spinner("Exporting analysis..."):
                try:
                    tokens_data, _, _, _, _ = analyze_text(text_input)
                    
                    # Convert to expected format
                    analysis_results = []
                    for token_data in tokens_data:
                        analysis_results.append({
                            'FORM': token_data['Token'],
                            'LEMMA': token_data['Lemma'],
                            'UPOS': token_data['POS'],
                            'DEPREL': token_data['Dependency']
                        })
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"outputs/analysis_{timestamp}.csv"
                    export_user_input_analysis(text_input, analysis_results, output_path)
                    
                    st.success(f"Analysis exported to {output_path}")
                    
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
    
    with col2:
        if st.button("Export Confusion Matrix to CSV"):
            with st.spinner("Exporting confusion matrix..."):
                try:
                    dataset_path = "dataset/UD_English-EWT/en_ewt-ud-train.conllu"
                    matrix = build_confusion_matrix(dataset_path)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"outputs/confusion_matrix_{timestamp}.csv"
                    save_confusion_csv(matrix, output_path)
                    
                    st.success(f"Confusion matrix exported to {output_path}")
                    
                except FileNotFoundError:
                    st.error("UD dataset not found. Please ensure the dataset is properly downloaded.")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
    
    st.divider()
    
    # Show existing plots
    st.header("Generated Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("outputs/accuracy.png"):
            st.subheader("Accuracy Plot")
            st.image("outputs/accuracy.png")
        else:
            st.info("No accuracy plot available. Run evaluation first.")
    
    with col2:
        if os.path.exists("outputs/confusion_matrix.png"):
            st.subheader("Confusion Matrix Heatmap")
            st.image("outputs/confusion_matrix.png")
        else:
            st.info("No confusion matrix available. Generate it first.")


if __name__ == "__main__":
    main()
