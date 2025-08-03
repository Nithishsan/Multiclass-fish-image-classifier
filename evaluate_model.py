import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model = tf.keras.models.load_model('models/resnet_fish_model.h5')

gen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
val_gen = gen.flow_from_directory('fish_dataset', target_size=(224,224),
    batch_size=8, class_mode='categorical', subset='validation', shuffle=False)

y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes
labels = list(val_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
