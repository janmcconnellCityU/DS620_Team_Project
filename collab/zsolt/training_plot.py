# collab/zsolt/training_plot.py
import json
import matplotlib.pyplot as plt

with open('checkpoints/zsolt_cnn_history.json', 'r', encoding='utf-8') as f:
    history = json.load(f)

if isinstance(history, list):
    # [{epoch, train_loss, val_loss, train_acc/accuracy, val_acc/accuracy}, ...]
    epochs     = [e.get('epoch', i + 1) for i, e in enumerate(history)]
    train_loss = [e['train_loss'] for e in history]
    val_loss   = [e['val_loss']   for e in history]
    train_acc  = [e.get('train_acc', e.get('train_accuracy')) for e in history]
    val_acc    = [e.get('val_acc',   e.get('val_accuracy'))   for e in history]
else:
    # {"train_loss":[...], "val_loss":[...], "train_acc":[...], "val_acc":[...]}
    epochs     = range(1, len(history['train_loss']) + 1)
    train_loss = history['train_loss']
    val_loss   = history['val_loss']
    train_acc  = history['train_acc']
    val_acc    = history['val_acc']

plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss,   label='Validation Loss')
plt.plot(epochs, train_acc,  label='Train Accuracy')
plt.plot(epochs, val_acc,    label='Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.title('CNN - Training vs Validation Metrics')
plt.tight_layout()
plt.show()