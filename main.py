from lightfm import LightFM
from lightfm.evaluation import auc_score
import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm_dataset_helper.lightfm_dataset_helper import DatasetHelper
import re
import sys

Ratings = pd.read_json(sys.argv[1], lines=True)
ratings_id = Ratings['UserId'].unique()
Content = pd.read_json(sys.argv[2], lines=True)

Target = pd.read_csv(sys.argv[3])

ratings_id = Ratings['UserId'].unique()
ratings_dict = pd.DataFrame(ratings_id)
ratings_dict.insert(0, 'id', range(1, len(ratings_dict) + 1))
ratings_dict = dict(zip(ratings_dict[0], ratings_dict['id']))

content_id = Content['ItemId'].unique()
content_dict = pd.DataFrame(content_id)
content_dict.insert(0, 'id', range(1, len(content_dict) + 1))

content_dict = dict(zip(content_dict[0], content_dict['id']))

dataset = Dataset()
dataset.fit((ratings_id),
    (content_id))

num_users, num_items = dataset.interactions_shape()

def extract_first_rating_value(ratings_list):
    if ratings_list:
        return ratings_list[0]["Value"]
    else:
        return None
    
# Aplicar a função à coluna "Ratings" para criar uma nova coluna "FirstRatingValue"
Content['FirstRatingValue'] = Content['Ratings'].apply(extract_first_rating_value)

def extract_numbers(text):
    # Use uma expressão regular para encontrar todos os números na string
    numbers = re.findall(r'\b\d+\b', str(text))
    # Soma os números encontrados (se houver algum)
    return sum(map(int, numbers)) if numbers else 0

# Aplicar a função à coluna "Awards" para criar uma nova coluna "QtdAwards"
Content['QtdAwards'] = Content['Awards'].apply(extract_numbers)

dataset.fit_partial(items=Content['ItemId'],
                    item_features=(Content['Metascore']))
dataset.fit_partial(items=Content['ItemId'],
                    item_features=(Content['imdbRating']))
dataset.fit_partial(items=Content['ItemId'],
                    item_features=(Content['Genre']))
dataset.fit_partial(items=Content['ItemId'],
                    item_features=(Content['Director']))
dataset.fit_partial(items=Content['ItemId'],
                    item_features=(Content['FirstRatingValue']))
dataset.fit_partial(items=Content['ItemId'],
                    item_features=(Content['QtdAwards']))

num_users, num_items = dataset.interactions_shape()

tuple_interactions = Ratings[['UserId', 'ItemId', 'Rating']]
tuple_interactions = list(tuple_interactions.to_records(index=False))

(interactions, weights) = dataset.build_interactions(tuple_interactions)

tuple_content = Content[["ItemId", "Metascore", "imdbRating", "Genre", "Director", "FirstRatingValue", "QtdAwards"]]
tuple_content = [(row['ItemId'], [row["Metascore"],  row["imdbRating"], row["Genre"], row["Director"], row["FirstRatingValue"], row["QtdAwards"]]) for _, row in tuple_content.iterrows()]

item_feat = dataset.build_item_features(tuple_content, normalize =True)

model = LightFM(loss='warp',
                learning_rate=0.005,
                learning_schedule='adagrad',
                no_components=5)

model.fit(interactions,
          sample_weight=weights ,
          item_features=item_feat,
          epochs=10)

def getItemModelId(userId, Target):
    data = []
    usar = Target.loc[Target['UserId'] == userId].copy()
    usar = usar['ItemId'].copy()
    for item in usar:
        itemId = content_dict.get(item) - 1
        data.append(int(itemId))

    return data

targets = Target['UserId'].unique()

print('UserId,ItemId')
for target in targets:
    customer_id = ratings_dict.get(target) - 1
    item_ids_cont = getItemModelId(target, Target)
    predictions = model.predict(customer_id, item_ids=item_ids_cont,item_features=item_feat, num_threads=4)
    top_items = sorted(zip(item_ids_cont, predictions), key=lambda x: -x[1])[:100]
    for rec in top_items:
        print(f'{target},{content_id[rec[0]]}')