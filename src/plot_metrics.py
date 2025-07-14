import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score


def plot_confusion_matrix(cm: np.ndarray, class_names: list = ["Negativo", "Positivo"], title: str = "Matriz de Confusión", cmap: str = "Blues", figsize: tuple = (6, 6), fontsize: int = 16, numberfmt: str = "d"):
    """Dibuja matriz de confusión con anotaciones."""
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=fontsize + 2, pad=12)
    ax.set_xlabel("Predicción", fontsize=fontsize, labelpad=9)
    ax.set_ylabel("Etiqueta Verdadera", fontsize=fontsize, labelpad=9)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)

    plt.setp(ax.get_yticklabels(), rotation=90, va= "center")

    fmt = "{:" + numberfmt + "}"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, fmt.format(cm[i, j]),
                    ha="center", va="center",
                    fontsize=fontsize, color=color)

    plt.tight_layout()
    plt.show()
    
    
def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, figsize: tuple[float, float] = (6, 4), lw: float = 3, font_size: int = 14, roc_color: str = 'teal', pr_color: str = 'orchid') -> None:
    """Dibuja matriz de confusión con anotaciones."""
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color=roc_color, lw=lw, 
             label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlabel("FPR", fontsize=font_size)
    plt.ylabel("TPR (Recall)", fontsize=font_size)
    plt.title("Curva ROC", fontsize=font_size + 2)
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color=pr_color, lw=lw)
    plt.xlabel("Recall", fontsize=font_size)
    plt.ylabel("Precision", fontsize=font_size)
    plt.title("Curva Precision-Recall", fontsize=font_size + 2)
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
def plot_training_losses(train_recon_losses: list[float], train_kl_losses: list[float], train_total_losses: list[float], figsize: tuple[float, float] = (6, 4), lw: float = 2.5, font_size: int = 14, recon_color: str = 'salmon', kl_color: str = 'seagreen', total_color: str = 'cornflowerblue') -> None:
    """Muestra pérdidas Recon, KL y Total por época."""
    
    epochs = range(1, len(train_recon_losses) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs, train_total_losses, color=total_color, lw=3.5, label='Total Loss (ELBO)')
    plt.plot(epochs, train_recon_losses, color=recon_color, lw=lw, label='Recon Loss (MSE)')
    plt.plot(epochs, train_kl_losses,    color=kl_color,    lw=lw, label='KL Loss')
    
    
    plt.title('Evolución de las Pérdidas durante el Entrenamiento', fontsize=font_size + 2, pad=12)
    plt.xlabel('Época', fontsize=font_size)
    plt.ylabel('Valor de Pérdida', fontsize=font_size)
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.legend(fontsize=font_size)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()