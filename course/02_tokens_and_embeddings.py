"""
Lesson 2 — Words to numbers (tokens and embeddings)

A computer doesn't understand "dog". It understands numbers.

To make text usable, we do two things:

  1. TOKENISE — give each word a unique ID (just look it up in a table).
  2. EMBED    — turn each ID into a vector of numbers (a row in another table).

Why two steps? IDs alone are meaningless — "dog" might be 5 and "cat" might be 6,
but 5 and 6 don't tell us anything about how those words relate.

An EMBEDDING is a vector of numbers — like coordinates in space. After training,
words with similar meanings end up at nearby points. "Dog" and "puppy" become
close; "dog" and "spaceship" become far apart. The model figures out where to
put each word on its own.

Run:  python3 02_tokens_and_embeddings.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

# ----------------------------------------------------------------------------
# Step 1: TOKENISE
# ----------------------------------------------------------------------------
# A vocabulary is just a list of all the words the model knows.
# Each word's position in the list is its ID.
vocab = ["<pad>", "<mask>", "dog", "cat", "fish", "bark", "meow", "swim"]
tok2id = {w: i for i, w in enumerate(vocab)}

print("Vocabulary:")
for i, w in enumerate(vocab):
    print(f"  id={i}  word={w}")

# Encoding a sentence = look up each word's ID.
sentence = ["dog", "bark"]
ids = [tok2id[w] for w in sentence]
print(f"\nSentence {sentence}  ->  ids {ids}")

# Note the two special tokens:
#   <pad>   — used to make all sequences in a batch the same length.
#   <mask>  — used in the fill-in-the-blank training game (Lesson 4).

# ----------------------------------------------------------------------------
# Step 2: EMBED
# ----------------------------------------------------------------------------
# An embedding table is just a 2-D table. Row i = vector for word i.
# We pick how many numbers each vector has — here, 4.
embedding_dim = 4
emb = nn.Embedding(len(vocab), embedding_dim)

print(f"\nEmbedding table shape: {tuple(emb.weight.shape)}")
print(f"  ({len(vocab)} words in vocab, each represented by {embedding_dim} numbers)")

# Looking up a word's vector is just indexing the table.
print("\nWord -> vector:")
for word in ["dog", "cat", "fish"]:
    i = tok2id[word]
    v = emb(torch.tensor(i)).detach()
    print(f"  {word:5s} ->  {v}")

# These vectors are RANDOM right now! Embeddings only become meaningful
# during training. The model nudges them so words used in similar contexts
# end up near each other.

# ----------------------------------------------------------------------------
# Encoding a whole sentence at once
# ----------------------------------------------------------------------------
ids_tensor = torch.tensor(ids)
vectors = emb(ids_tensor)
print(f"\nSentence vectors:  shape {tuple(vectors.shape)}  (2 words, 4 numbers each)")
print(vectors.detach())

# Measuring "similarity" between words: the dot product. Two vectors that
# point in the same direction have a big dot product. Two random vectors
# usually have a small one.
def similarity(a, b):
    return torch.dot(emb(torch.tensor(tok2id[a])), emb(torch.tensor(tok2id[b]))).item()

print("\nSimilarity (random embeddings, all meaningless for now):")
print(f"  dog · cat   = {similarity('dog', 'cat'):.2f}")
print(f"  dog · fish  = {similarity('dog', 'fish'):.2f}")
print(f"  bark · meow = {similarity('bark', 'meow'):.2f}")
# After training in Lesson 4, these numbers would start to reflect real
# relationships — e.g., bark and meow would be similar because they're both
# "sounds an animal makes".

# ----------------------------------------------------------------------------
# Exercises
# ----------------------------------------------------------------------------
# 1. Change embedding_dim to 16. What's the new shape of the embedding table?
#
# 2. Print the vector for "<mask>". It's random too — but after training it
#    will end up at a "neutral" point because it has no inherent meaning.
#
# 3. Add three new words to the vocabulary: "bird", "tweet", "fly".
#    Encode the sentence ["bird", "tweet"] and print its vectors.
#
# 4. (Stretch) Write a tiny loop that finds, for each word, the OTHER word
#    in the vocab with the highest dot product. Right now the answers will
#    be random — but you've built the tool that lets us find "nearest
#    neighbours" once training has shaped the embeddings.
