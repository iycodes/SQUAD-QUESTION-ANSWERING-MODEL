from flask import Flask, request, jsonify
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your saved model and pretrained tokenizer
model_directory = "./tinybert_squad_final"
model = AutoModelForQuestionAnswering.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding='max_length', max_length=384)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index]))
    return answer

def compute_similarity(answer1, answer2):
    vectorizer = TfidfVectorizer().fit_transform([answer1, answer2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    question = data['question']
    context = data['context']
    user_answer = data['user_answer']

    model_answer = get_answer(question, context)
    similarity = compute_similarity(model_answer, user_answer)

    return jsonify({
        'model_answer': model_answer,
        'user_answer': user_answer,
        'similarity_score': similarity * 100  # Convert to percentage
    })

if __name__ == '__main__':
    app.run(debug=True)
