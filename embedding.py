import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the sentence dataset
df = pd.read_csv("Data/text_generation_results.csv")
# Get sentences
sentences = df["sentence"].tolist()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(sentences)

# Save embeddings
np.save("data/embeddings.npy", embeddings)

print("Embeddings generated successfully!")
print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between all sentence embeddings
similarity_matrix = cosine_similarity(embeddings)

# Store all unique sentence pairs
results = []

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        results.append({
            "Text 1": sentences[i],
            "Text 2": sentences[j],
            "Cosine Similarity": round(similarity_matrix[i][j], 4)
        })

# Convert results into DataFrame
results_df = pd.DataFrame(results)

# Sort from highest similarity to lowest
results_df = results_df.sort_values(
    by="Cosine Similarity",
    ascending=False
)

# Save similarity results
results_df.to_csv(
    "Data/similarity_results.csv",
    index=False
)

print("\nComparison completed!")
print("\nTop 5 most similar pairs:")
print(results_df.head(5))
# Save top 10 sentence pairs
top_10 = results_df.head(10)
top_10.to_csv("Data/top_10_pairs.csv", index=False)

print("\nTop 10 pairs saved successfully!")