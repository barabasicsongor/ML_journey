import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)

# Weighted approximate-rank pairwise
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("\nUser {}".format(user_id))
        print("Known Positives:")

        for x in known_positives[:3]:
            print(x)
        
        print("\nRecommended:")

        for x in top_items[:3]:
            print(x)

sample_recommendation(model, data, [3, 25, 450])