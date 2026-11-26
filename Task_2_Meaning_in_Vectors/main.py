
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What is the leave policy?",
    "How many casual leaves do I get?",
    "The weather in Chennai is hot",
    "Office timings are 9 to 6"
]

print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = model.encode(sentences, convert_to_tensor=True)

print("\nTASK 2: MEANING IN VECTORS")
print("Embedding shape:", tuple(vectors.shape))

similarity = util.cos_sim(vectors, vectors).cpu().numpy()

print("\nCOSINE SIMILARITY MATRIX")

print(np.round(similarity, 3))

assert similarity.shape == (6, 6)

pairs = [
    (float(similarity[i, j]), i, j)
    for i, j in itertools.combinations(range(len(sentences)), 2)
]

pairs.sort(reverse=True)

print("\nTOP 3 MOST SIMILAR PAIRS")

for score, i, j in pairs[:3]:
    print(f"{score:.3f} | {sentences[i]!r} <-> {sentences[j]!r}")

score, i, j = pairs[-1]

print("\nLEAST SIMILAR PAIR")
print(f"{score:.3f} | {sentences[i]!r} <-> {sentences[j]!r}")

a, b = 0, 1

words_a = set(sentences[a].lower().replace("?", "").split())
words_b = set(sentences[b].lower().replace("?", "").split())

print("\nSIMILAR MEANING, DIFFERENT WORDS")
print("Candidate pair:", sentences[a], "<->", sentences[b])
print("Cosine similarity:", round(float(similarity[a, b]), 3))
print("Shared words:", sorted(words_a & words_b))

points_2d = PCA(
    n_components=2,
    random_state=42
).fit_transform(vectors.cpu().numpy())

plt.figure(figsize=(10, 7))
plt.scatter(points_2d[:, 0], points_2d[:, 1])

for i, label in enumerate(sentences):
    plt.annotate(
        label,
        (points_2d[i, 0], points_2d[i, 1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.title("Sentence Embeddings Reduced to 2D with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("task2_pca_plot.png", dpi=300)
plt.show()

print("\nPCA graph saved as task2_pca_plot.png")
print("\nTASK 2 COMPLETED SUCCESSFULLY!")
