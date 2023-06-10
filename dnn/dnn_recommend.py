import pickle
import pandas as pd
import torch 
from dnn_model import DNN

def get_dnn_recommendations(userId): 
	
	userId = int(userId)

	loaded_model = pickle.load(open('./dnn/dnn_model.sav', 'rb'))
	movie_ids_map = pickle.load(open('./dnn/movie_ids_map.pkl', 'rb'))
	user_ids_map = pickle.load(open('./dnn/user_ids_map.pkl', 'rb'))
	movie_dict = pickle.load(open('./dnn/movie_dict.pkl', 'rb'))

	unseen = movies_unseen_by_user(userId)
	unseen_encoded_ids = [movie_ids_map[m] for m in unseen]
	
	user_id_encoded = user_ids_map[userId]

	predictions = [loaded_model(torch.tensor([user_id_encoded]), torch.tensor([m])) for m in unseen_encoded_ids]
	preds = [p.item() for p in predictions]
	
	top_preds = sorted(range(len(preds)), key=lambda i: preds[i])[-10:]
	pred_ratings = [round(preds[p] -.5, 2) if preds[p] - .5<5 else 5.0 for p in top_preds]
	top_movies = [movie_dict[unseen[p]] for p in top_preds]
	
	result = list(zip(top_movies, str(pred_ratings))
	# print(result)
	return result


def movies_unseen_by_user(userId):
  data = pd.read_csv('./dnn/ml-latest-small/ratings.csv', parse_dates=['timestamp'])
  user_data = data[data['userId']==userId]
  seen = set(user_data['movieId'])
  total = set(data['movieId'])
  unseen = total.difference(seen)
  return list(unseen)


