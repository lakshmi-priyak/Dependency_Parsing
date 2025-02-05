import spacy # type: ignore
from spacy.tokens import Doc # type: ignore
from spacy.training import Example # type: ignore
import random
import os

# Load a blank English model
nlp = spacy.blank("en")

# Add a dependency parser
if "parser" not in nlp.pipe_names:
    parser = nlp.add_pipe("parser", last=True)

# Define dependency labels
for label in ["nsubj", "ROOT", "dobj"]:
    parser.add_label(label)

# Training data
TRAIN_DATA = [
    ("I love NLP.", {"words": ["I", "love", "NLP", "."], "heads": [1, 1, 1, 1], "deps": ["nsubj", "ROOT", "dobj", "punct"]}),
    ("She plays soccer.", {"words": ["She", "plays", "soccer", "."], "heads": [1, 1, 1, 1], "deps": ["nsubj", "ROOT", "dobj", "punct"]}),
    ("He studies AI.", {"words": ["He", "studies", "AI", "."], "heads": [1, 1, 1, 1], "deps": ["nsubj", "ROOT", "dobj", "punct"]}),
]

# Convert data to spaCy Example format
examples = []
for text, annotations in TRAIN_DATA:
    words = annotations["words"]
    heads = annotations["heads"]
    deps = annotations["deps"]
    doc = Doc(nlp.vocab, words=words)
    example = Example.from_dict(doc, {"heads": heads, "deps": deps})
    examples.append(example)

# Train the model
optimizer = nlp.initialize()
for epoch in range(50):
    random.shuffle(examples)
    losses = {}
    nlp.update(examples, drop=0.5, losses=losses)
    print(f"Epoch {epoch + 1}, Losses: {losses}")

# Save the model
MODEL_PATH = os.path.join(os.getcwd(), "dependency_parser_model")
nlp.to_disk(MODEL_PATH)
print(f"Model saved successfully at: {MODEL_PATH}")
