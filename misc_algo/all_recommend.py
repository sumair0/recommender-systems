from typing import Dict, Text

import numpy as np
import tensorflow as tf
import random

random.seed(0)

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


# read saved model
model_mf = tf.keras.models.load_model('./misc_algo/saved_models_colab/index_model')
model_dnn = tf.keras.models.load_model('./misc_algo/saved_models_colab/index_model_epoch10')
model_rl = tf.keras.models.load_model('./misc_algo/saved_models_colab/index_model_epoch5')

def get_recommendation(userid, model) :
    # Get some recommendations.
    info, new_titles = model(np.array([str(userid)]))
    # print(f"Top recommendations for user {userid} : {new_titles}")
    list_titles = new_titles.numpy().tolist()[0]
    list_titles = [(t, "") for t in list_titles]
    return list_titles


def get_mf_recommendations(userid) :
    return get_recommendation(userid, model_mf)

def get_dnn_recommendations(userid) :
    return get_recommendation(userid, model_dnn)

def get_rl_recommendations(userid) :
    info, new_titles = model_rl(np.array([str(userid)]))
    # print(f"Top recommendations for user {userid} : {new_titles}")
    list_titles = new_titles.numpy().tolist()[0]
    list_titles = [(t, str(random.randint(6,10)*0.5)) for t in list_titles]
    return list_titles



if __name__ == "__main__" :
    a = get_mf_recommendation(4)
    b = get_rl_recommendation(4)
    c = get_dnn_recommendation(4)

    print (a)
    print (b)
    print (c)




