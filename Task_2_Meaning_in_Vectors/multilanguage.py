
import numpy as np
from sentence_transformers import SentenceTransformer, util

sentences = [
    "I want to reset my password.",
    "எனது கடவுச்சொல்லை மீட்டமைக்க வேண்டும்.",
    "मुझे अपना पासवर्ड रीसेट करना है।",
    "The weather is very hot today.",
    "இன்று வானிலை மிகவும் சூடாக உள்ளது."
]

print("Loading multilingual embedding model...")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings = model.encode(sentences, convert_to_tensor=True)

print("MULTILINGUAL EMBEDDING TEST")
print("Embedding shape:", tuple(embeddings.shape))

similarity = util.cos_sim(embeddings, embeddings).cpu().numpy()

print("COSINE SIMILARITY MATRIX")
print(np.round(similarity, 3))

print("SIMILARITY WITH ENGLISH PASSWORD SENTENCE")

for i in range(1, 3):
    score = similarity[0][i]
    print(sentences[i])
    print("Similarity:", round(float(score), 3))

print("SIMILARITY WITH ENGLISH WEATHER SENTENCE")
print(sentences[3])
print("Tamil:", sentences[4])
print("Similarity:", round(float(similarity[3][4]), 3))

print("MULTILINGUAL TEST COMPLETED SUCCESSFULLY!")
