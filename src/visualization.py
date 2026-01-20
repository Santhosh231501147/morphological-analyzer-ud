import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def ensure_outputs_folder():
    """Create outputs folder if it doesn't exist."""
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    return outputs_dir


def plot_accuracy_bar(accuracy):
    """
    Plot accuracy as a bar chart and save as PNG.
    
    Args:
        accuracy (float): Accuracy value between 0 and 1
    """
    outputs_dir = ensure_outputs_folder()
    
    plt.figure(figsize=(8, 6))
    plt.bar(['POS Tagging Accuracy'], [accuracy], color='skyblue', edgecolor='navy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('POS Tagging Accuracy')
    plt.grid(axis='y', alpha=0.3)
    
    # Add accuracy value on top of bar
    plt.text(0, accuracy + 0.02, f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_heatmap(matrix):
    """
    Plot confusion matrix as heatmap and save as PNG.
    
    Args:
        matrix (dict): Confusion matrix dictionary [gold_pos][pred_pos] = count
    """
    outputs_dir = ensure_outputs_folder()
    
    # Get all unique POS tags
    all_pos_tags = set()
    for gold_pos, predictions in matrix.items():
        all_pos_tags.add(gold_pos)
        for pred_pos in predictions:
            all_pos_tags.add(pred_pos)
    
    # Sort POS tags for consistent ordering
    pos_tags = sorted(all_pos_tags)
    
    # Create matrix array
    matrix_array = np.zeros((len(pos_tags), len(pos_tags)))
    
    for i, gold_pos in enumerate(pos_tags):
        for j, pred_pos in enumerate(pos_tags):
            matrix_array[i, j] = matrix.get(gold_pos, {}).get(pred_pos, 0)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix_array, 
                xticklabels=pos_tags, 
                yticklabels=pos_tags,
                annot=True, 
                fmt='g',
                cmap='Blues',
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted POS')
    plt.ylabel('Gold POS')
    plt.title('POS Tagging Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
