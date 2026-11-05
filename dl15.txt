1from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
corpus = [
    ["deep", "learning", "is", "fun"],
    ["machine", "learning", "uses", "data"],
    ["deep", "neural", "networks"],
    ["artificial", "intelligence", "uses", "learning"],
    ["python", "is", "easy"]
]
model = Word2Vec(corpus, vector_size=20, window=2, min_count=1)
print("Embedding of 'learning':")
print(model.wv["learning"])
print("\nSimilar words to 'learning':")
print(model.wv.most_similar("learning"))
words = list(model.wv.index_to_key)
vectors = [model.wv[word] for word in words]
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)
for i, word in enumerate(words):
    plt.scatter(result[i, 0], result[i, 1])
    plt.text(result[i, 0], result[i, 1], word)
plt.title("Word Embeddings")
plt.show()



#===============


 
5 import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
 (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000)
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=32),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(46, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print("\nDocument\tActual\tPredicted")
for i in range(10):
    print(f"{i+1}\t\t{y_test[i]}\t{predicted_classes[i]}")
