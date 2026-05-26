"""
Lesson 2 — Words to numbers (tokens and embeddings)

A computer doesn't understand "dog". It understands numbers.

To make text usable, we do two things:

  1. TOKENISE — give each word a unique ID (just look it up in a table).
  2. EMBED    — turn each ID into a vector of numbers (a row in another table).

Why two steps? IDs alone are meaningless — "dog" might be 5 and "cat" might be 6,
but 5 and 6 don't tell us anything about how those words relate.

An EMBEDDING is a list of numbers that describes a word — like a checklist of
"hidden features". Think of how you might describe a person with 4 numbers
(0-10 for: ice cream, soccer, video games, cleaning). Two kids with similar
lists are similar people. Words work the same way: similar words end up with
similar number lists. The computer makes up the "features" by itself during
training — nobody tells it what each slot means.

To compare two lists, use a DOT PRODUCT — multiply matching slots, add the
products. Big total = similar. Small total = not similar. We do an example
by hand below.

For the full intuition with worked examples, see the companion walkthrough:
  02_walkthrough.md  (read it before reading the code if any of this is fuzzy)

Run:  python3 02_tokens_and_embeddings.py
"""
import torch                                    # PyTorch: tensors, autograd.
import torch.nn as nn                           # nn.Embedding lives here.

torch.manual_seed(0)                            # Reproducible RNG.
torch.set_printoptions(precision=2, sci_mode=False)     # Clean tensor printing.

# ----------------------------------------------------------------------------
# Step 1: TOKENISE
# ----------------------------------------------------------------------------
# A vocabulary is just a list of all the words the model knows.
# Each word's position in the list is its ID.
vocab = ["<pad>", "<mask>", "dog", "cat", "fish", "bark", "meow", "swim"]   # 8 tokens.
tok2id = {w: i for i, w in enumerate(vocab)}    # Lookup table: word → integer id.

print("Vocabulary:")
for i, w in enumerate(vocab):                   # Iterate paired (id, word).
    print(f"  id={i}  word={w}")

# Encoding a sentence = look up each word's ID.
sentence = ["dog", "bark"]                      # Two-word example sentence.
ids = [tok2id[w] for w in sentence]             # Convert each word to its id.
print(f"\nSentence {sentence}  ->  ids {ids}")

# Note the two special tokens:
#   <pad>   — used to make all sequences in a batch the same length.
#   <mask>  — used in the fill-in-the-blank training game (Lesson 4).

# ----------------------------------------------------------------------------
# Step 2: EMBED
# ----------------------------------------------------------------------------
# An embedding table is just a 2-D table. Row i = vector for word i.
# Same idea as the "kid checklist" in the intro: each word is described by
# 4 numbers (hidden features the computer makes up itself during training).
# Real models use 512, 768, or 1024 numbers per word — we use 4 so we can
# print them and look at them.
embedding_dim = 4                               # How many numbers per word.
emb = nn.Embedding(len(vocab), embedding_dim)   # 8×4 trainable lookup table.

print(f"\nEmbedding table shape: {tuple(emb.weight.shape)}")    # Should print (8, 4).
print(f"  ({len(vocab)} words in vocab, each represented by {embedding_dim} numbers)")

# Looking up a word's vector is just indexing the table.
print("\nWord -> vector:")
for word in ["dog", "cat", "fish"]:             # Show 3 example words.
    i = tok2id[word]                            # Find the id.
    v = emb(torch.tensor(i)).detach()           # Look up row in table; detach from autograd.
    print(f"  {word:5s} ->  {v}")

# These vectors are RANDOM right now! Embeddings only become meaningful
# during training. The model nudges them so words used in similar contexts
# end up near each other.

# ----------------------------------------------------------------------------
# Encoding a whole sentence at once
# ----------------------------------------------------------------------------
ids_tensor = torch.tensor(ids)                  # Convert id list to tensor.
vectors = emb(ids_tensor)                       # Look up all words at once. Shape (2, 4).
print(f"\nSentence vectors:  shape {tuple(vectors.shape)}  (2 words, 4 numbers each)")
print(vectors.detach())

# Measuring "similarity" between two words: the dot product.
# Same math as the kid example from the intro: multiply matching slots
# (slot 1 × slot 1, slot 2 × slot 2, ...) and add the products.
# Big number = words are similar. Small or negative = words are different.
# (Right now the embeddings are random, so these numbers are meaningless.
#  After training, they would actually reflect word meaning.)
def similarity(a, b):
    # Compute dot product between embeddings of words a and b.
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
