import argparse

from flask import Flask, render_template, request

from scripts.utils import load_model, get_processors, TextSummariser

app = Flask(__name__)


@app.route('/', methods=['post', 'get'] )
def index():
    summary = None
    if request.method == 'POST':
        summary = input_processor(request.form.get('text'))
    return render_template('index.html', summary=summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, required=True)

    args = parser.parse_args()
    processors = get_processors(args.embedding_model)
    model = load_model(args.model_path)
    input_processor = TextSummariser(processors, model)
    app.run(host='0.0.0.0', port=5000)
