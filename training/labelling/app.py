# -*- coding: utf-8 -*-
import os

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
DATA_FOLDER = 'static'
TRAINING_SET = [i.split('.')[0] for i in os.listdir(DATA_FOLDER) if i.endswith('.png')]


@app.route('/', methods=['GET', 'POST'])
def index():
    idx = int(request.args.get('index', 0))
    if idx < 0 or idx >= len(TRAINING_SET):
        return 'image index out of bounds', 404

    current = TRAINING_SET[idx]
    img_path = f'{DATA_FOLDER}/{current}.png'
    txt_path = f'{DATA_FOLDER}/{current}.gt.txt'

    # Save the labelled text
    if request.method == 'POST':
        text = request.form['edited_text']
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return redirect(url_for('index', index=idx + 1))

    with open(txt_path, 'r', encoding='utf-8') as f:
        default_text = f.read()

    return render_template('index.html', img_path=img_path, text=default_text, index=idx)


if __name__ == '__main__':
    app.run(debug=True)
