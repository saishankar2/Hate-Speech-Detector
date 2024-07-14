import gradio as gr
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
# Load the pre-trained model
model = load_model('toxicity.h5')
# Load the TextVectorization layer
vectorizer = TextVectorization(max_tokens=200000,
output_sequence_length=1800,
output_mode='int')
# Load the vocabulary from your training data
df = pd.read_csv('train.csv')
X = df['comment_text']
vectorizer.adapt(X.values)
# Define a function to score comments
def score_comment(comment):
# Vectorize the comment
vectorized_comment = vectorizer([comment])
# Get the prediction
results = model.predict(vectorized_comment)
# Format the results
text = ''
for idx, col in enumerate(df.columns[2:]):
text += '{}: {}\n'.format(col, results[0][idx]>0.5)
return text
interface = gr.Interface(fn=score_comment,
inputs=gr.Textbox(lines=2, placeholder='Enter a comment'),
outputs='text',
title='Hate Speech Detector')
# Launch the interface
interface.launch()
