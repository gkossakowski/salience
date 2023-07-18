from flask import Flask, render_template, make_response
from .salience import extract, terminal_distr
import numpy as np

app = Flask(__name__)

@app.template_global()
def scale(score):
    return max(0, min(1, score ** 3 - 0.95))

with open('./transcript.txt', 'r') as file:
    source_text = file.read().strip()
sentence_ranges, adjacency = extract(source_text)
scores = terminal_distr(adjacency)

@app.route("/salience", methods=['GET'])
def salience_view():
    try:
        sentences = [(score, source_text[start:end]) for score, (start, end) in zip(scores, sentence_ranges)]
        return render_template('salience.html', sentences=sentences)
    except Exception as e:
        return make_response({'error': str(e)}, 500)
