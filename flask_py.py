

from pandas import DataFrame
from flask import Flask,request,jsonify, make_response
import tensorflow as tf
import json
# import warnings
# warnings.filterwarnings("ignore")
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import set_session

from utils1 import *

application = Flask(__name__)  


MODEL_THRESHOLD = 0.8
MODEL_WEIGHTS_FILE = 'model_weights.h5'

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

set_session(sess)
model = tf.keras.models.load_model(MODEL_WEIGHTS_FILE)


@application.route('/get_sentence_similarity', methods = ['POST'])
def sentenceSimilarity():
    global model, graph, sess
    
    if 'injson' not in request.form:
        return make_response(jsonify([{"error": "Request body absent"}]), 400)
    
    try:
        json_data = json.loads(request.form['injson'])
    except Exception as e:
        return make_response(jsonify([{"error": "Error parsing JSON"}]), 400)
    
    if 'sentence1' not in json_data or 'sentence2' not in json_data:
        return make_response(jsonify([{"error": "Invalid JSON"}]), 400)
    
    data = DataFrame(json_data, index=[0])
    sentence1_test, sentence2_test = get_list(data)
    sentence1_test = remove_punctuation(sentence1_test)
    sentence2_test = remove_punctuation(sentence2_test)

    sentence1_word_sequences_test, sentence2_word_sequences_test = my_tokenizer(sentence1_test, sentence2_test)

    q1_data_test, q2_data_test = finaL_save(sentence1_word_sequences_test, sentence2_word_sequences_test)
    

    with graph.as_default():
        set_session(sess)
        try:
            y_pred = model.predict([q1_data_test, q2_data_test])
        except Exception as e:
            return make_response(jsonify([{"error": "Model prediction failed"}]), 500)    
    
    
    y_pred = np.where(y_pred > MODEL_THRESHOLD, 1, 0)
    
    json_data['isSimilar'] = int(y_pred[0])
    
    return make_response(jsonify([json_data]), 200)


if __name__ == '__main__':
    application.run(host= '0.0.0.0',port=4002)
