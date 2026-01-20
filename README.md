# Morphological Analyzer

A comprehensive morphological analysis tool for English text using spaCy and Universal Dependencies.

## Features

- **Text Analysis**: Morphological analysis of user input text
- **UD Dataset Processing**: Load and analyze Universal Dependencies datasets
- **POS Tagging Evaluation**: Evaluate accuracy against gold standard annotations
- **Confusion Matrix**: Generate and visualize POS tagging confusion matrices
- **Multiple Interfaces**: CLI, API, and GUI interfaces
- **Export Options**: CSV export for analysis results and confusion matrices
- **Visualization**: Generate accuracy plots and confusion matrix heatmaps
- **Batch Processing**: Process multiple text files efficiently

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

3. Download UD English-EWT dataset and place in `dataset/UD_English-EWT/`:
   - `en_ewt-ud-train.conllu`
   - `en_ewt-ud-dev.conllu`
   - `en_ewt-ud-test.conllu`

## Usage

### Command Line Interface

Run the main CLI program:
```bash
python main.py
```

Options:
1. User input analysis
2. UD dataset analysis
3. POS accuracy evaluation
4. Generate confusion matrix
5. Export CSV
6. Batch processing
7. Start REST API
8. Start GUI
0. Exit

### Accuracy Evaluation

Evaluate POS tagging accuracy on UD dataset:

**Via CLI:**
```bash
python main.py
# Select option 3: POS accuracy evaluation
```

**Via API:**
```bash
curl -X GET "http://localhost:8000/evaluate"
```

**Via GUI:**
- Launch GUI and click "Run Accuracy Evaluation" button

**Output:**
- Overall accuracy percentage
- Total tokens processed
- Correct predictions count
- Accuracy plot saved to `outputs/accuracy.png`

### Confusion Matrix

Generate POS tagging confusion matrix:

**Via CLI:**
```bash
python main.py
# Select option 4: Generate confusion matrix
```

**Via API:**
```bash
curl -X GET "http://localhost:8000/confusion-matrix"
```

**Via GUI:**
- Launch GUI and click "Generate Confusion Matrix" button

**Output:**
- Confusion matrix in JSON format
- CSV export: `outputs/confusion_matrix_TIMESTAMP.csv`
- Heatmap visualization: `outputs/confusion_matrix.png`

### Batch Processing

Process multiple text files:

**Via CLI:**
```bash
python main.py
# Select option 6: Batch processing
```

**Direct command:**
```bash
# Process single file
python src/batch_processor.py --input file.txt --output results.csv

# Process all files in datasets/ directory
python src/batch_processor.py --datasets
```

**Supported formats:**
- Input: `.txt` files (one sentence per line)
- Output: CSV with morphological analysis

### REST API Usage

Start the API server:
```bash
python src/api.py
```

Or run with uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**

**Analyze single sentence:**
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "The cat sits on the mat"}'
```

**Batch analysis:**
```bash
curl -X POST "http://localhost:8000/batch" \
     -H "Content-Type: application/json" \
     -d '{"sentences": ["The cat sits", "The dog runs"]}'
```

**Get evaluation results:**
```bash
curl -X GET "http://localhost:8000/evaluate"
```

**Get confusion matrix:**
```bash
curl -X GET "http://localhost:8000/confusion-matrix"
```

**API Documentation:**
- Interactive docs: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/openapi.json`

### GUI Usage

Launch the Streamlit GUI:
```bash
streamlit run src/gui.py
```

**GUI Features:**
- **Text Input**: Enter text for morphological analysis
- **Token Table**: View detailed token analysis (form, lemma, POS, dependency)
- **Statistics**: Count nouns, verbs, and extract noun/verb phrases
- **Evaluation**: Run accuracy evaluation on UD dataset
- **Confusion Matrix**: Generate and visualize confusion matrix
- **Export**: Export analysis results and confusion matrices to CSV
- **Plots**: View generated accuracy plots and confusion matrix heatmaps

**GUI Workflow:**
1. Enter text in the input box
2. Click "Analyze Text" to see token analysis
3. Use evaluation buttons to test on UD dataset
4. Export results using export buttons
5. View generated plots in the Plots section

### CSV Export

Export analysis results to CSV format:

**Via CLI:**
```bash
python main.py
# Select option 5: Export CSV
```

**Export types:**
- User input analysis: `outputs/user_analysis_TIMESTAMP.csv`
- UD dataset analysis: `outputs/ud_analysis_TIMESTAMP.csv`
- Confusion matrix: `outputs/confusion_matrix_TIMESTAMP.csv`

**CSV Format:**
```csv
sentence,token,lemma,pos,dependency
"The cat sits","cat","cat","NOUN","nsubj"
"The cat sits","sits","sit","VERB","root"
```

### Visualization

Generate plots and visualizations:

**Accuracy Plot:**
- Bar chart showing overall POS tagging accuracy
- Saved as `outputs/accuracy.png`
- Generated automatically during evaluation

**Confusion Matrix Heatmap:**
- Color-coded matrix of gold vs predicted POS tags
- Saved as `outputs/confusion_matrix.png`
- Shows misclassification patterns

**Viewing plots:**
- In GUI: Plots section displays generated visualizations
- Direct access: Files in `outputs/` directory

## Project Structure

```
morphological_analyzer/
├── main.py                 # CLI interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── ud_loader.py        # UD dataset loader
│   ├── evaluation.py       # POS evaluation and confusion matrix
│   ├── exporter.py         # CSV export functionality
│   ├── visualization.py    # Plot generation
│   ├── api.py             # FastAPI server
│   ├── gui.py             # Streamlit GUI
│   └── batch_processor.py # Batch processing
├── dataset/
│   └── UD_English-EWT/    # UD dataset files
├── datasets/              # Batch processing input files
├── outputs/               # Generated plots and exports
└── README.md
```

## Example Usage

### CLI Example
```bash
python main.py
# Select option 1
# Enter: "The quick brown fox jumps over the lazy dog"
# View morphological analysis
```

### API Example
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "The cat sits on the mat"}'
```

### Batch Processing Example
```bash
# Create datasets/sample.txt with sentences
echo "The cat sits on the mat" > datasets/sample.txt
echo "The dog runs in the park" >> datasets/sample.txt

# Process the file
python src/batch_processor.py --datasets
```

## Output Formats

### Token Analysis
Each token includes:
- **FORM**: Original word form
- **LEMMA**: Base form of the word
- **UPOS**: Universal POS tag
- **HEAD**: Head token index
- **DEPREL**: Dependency relation

### Evaluation Metrics
- **Accuracy**: Overall POS tagging accuracy
- **Confusion Matrix**: Detailed POS tag comparison
- **Visualizations**: Accuracy plots and heatmaps

## Dependencies

- spaCy: NLP processing and POS tagging
- FastAPI: REST API framework
- Streamlit: Web GUI framework
- matplotlib/seaborn: Data visualization
- pandas: Data manipulation
- NLTK: Natural language processing utilities
- uvicorn: ASGI server for FastAPI
