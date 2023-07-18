import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import nltk.data
import nltk
import openai

import os

env_openai_key = os.environ.get('OPENAI_API_KEY')

if not env_openai_key:
    raise Exception("OPENAI_API_KEY environment variable not set")

openai.api_key = env_openai_key

nltk.download('punkt')

model = SentenceTransformer('all-mpnet-base-v2')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def degree_power(A, k):
    # Ensure A is a numpy array before doing degree power
    if isinstance(A, torch.Tensor):
        A = A.numpy()

    degrees = np.power(A.sum(axis=1), k)
    D = np.diag(degrees)
    return D

# def normalized_adjacency(A):
#     # Ensure A is a numpy array before calculating normalized adjacency
#     if isinstance(A, torch.Tensor):
#         A = A.numpy()

#     D_inv_sqrt = degree_power(A, -0.5)
#     return torch.from_numpy(D_inv_sqrt @ A @ D_inv_sqrt)

def row_normalize_adjacency(A):
    row_sum = A.sum(axis=1, keepdims=True)
    return A / row_sum

def get_sentences(source_text):
    sentence_ranges = list(sent_detector.span_tokenize(source_text))
    sentences = [source_text[start:end] for start, end in sentence_ranges]
    return sentences, sentence_ranges

import re

def split_text_into_segments(text):
    # Split the text into segments
    segments = text.split("\n\n")
    
    segments_info = []
    ranges_info = []
    last_end_pos = -1
    for segment in segments:
        segment = segment.strip()
        if segment != '':  # ignore empty segments
            # Calculate start and end positions
            start_pos = text.find(segment, last_end_pos)
            end_pos = start_pos + len(segment)
            segments_info.append(segment)
            ranges_info.append((start_pos, end_pos))
            last_end_pos = end_pos
    
    return segments_info, ranges_info


def get_vectors(sentences):
    MAX_CHARS = 8000 * 4 # OpenAI API token limit, the limit is 8191 but we cap ourselves at 8000 to be safe
    # Splits sentences list into chunks of 8000 characters or less
    chunk = []
    embeddings = []

    def process_chunk(chunk):
        print("chunk size: ", len(chunk))
        print("Issuing OpenAI request")

        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        chunk_embeddings = [response["data"][i]["embedding"] for i in range(len(chunk))]
        embeddings.extend(chunk_embeddings)

    for sentence in sentences:
        if len(''.join(chunk)) + len(sentence) < MAX_CHARS:
            chunk.append(sentence)
        else:
            process_chunk(chunk)
            chunk = [sentence]
    
    process_chunk(chunk)

    return embeddings

def text_rank(sentences):
    vectors = model.encode(sentences)
    # vectors = get_vectors(sentences)
    vectors = torch.tensor(vectors)
    print("vectors shape: ", vectors.shape)
    dot_product = torch.matmul(vectors, vectors.T)
    vectors_norm = vectors.norm(dim=1).unsqueeze(1)
    adjacency = dot_product / (vectors_norm * vectors_norm.T)
    adjacency = torch.clamp(adjacency, min=0, max=1)
    adjacency.fill_diagonal_(0.)
    return row_normalize_adjacency(adjacency)

def terminal_distr(adjacency, initial=None):
    sample = initial if initial is not None else torch.full((adjacency.shape[0],), 1.)
    steps = 150
    scores = sample.matmul(torch.matrix_power(adjacency, steps)).tolist()
    return scores

def extract(source_text):
    # sentences, sentence_ranges = get_sentences(source_text)
    sentences, sentence_ranges = split_text_into_segments(source_text)
    print(len(source_text))
    print(sentence_ranges[-10:])
    print(len(sentences))
    adjacency = text_rank(sentences)
    return sentence_ranges, adjacency

def get_results(source_text):
    sentences, sentence_ranges = get_sentences(source_text)
    adjacency = text_rank(sentences)
    scores = terminal_distr(adjacency)
    for score, sentence in sorted(zip(scores, sentences), key=lambda xs: xs[0]):
        if score > 1.1:
            print('{:0.2f}: {}'.format(score, sentence))
