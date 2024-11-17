import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_histogram(series: pd.Series, value_label_dict: dict[int, str], title, filepath):
    bin_edges = sorted(value_label_dict.keys()) + [max(value_label_dict.keys()) + 1]
    bin_edges.sort()
    
    ax = series.hist(bins=bin_edges, align='left', edgecolor='black')    
    ax.set_xlabel('Category')
    ax.set_ylabel('Frequency')    
    
    plt.xticks(ticks=list(value_label_dict.keys()), labels=list(value_label_dict.values()))
    plt.title(title)    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_boxplot(series: pd.Series, title, filepath):
    plt.figure(figsize=(8, 6))
    plt.boxplot(series, medianprops=dict(color='red'))
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks([])
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_bar(words, counts, title, filepath):
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_simple(training_values, validation_values, test_value, xlabel, ylabel, filepath):
    plt.plot(training_values, label="training")
    plt.plot(validation_values, label="validation")
    plt.axhline(y=test_value, color='r', linestyle='--', label='test')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(training_values)))
    plt.legend()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, class_names, filepath):
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, 
                annot=True, 
                fmt=".1f", 
                cmap="Blues", 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.savefig(filepath)
    plt.show()   