from ItemKNN import ItemKNN
params = {"train_data": "../../data/AmazonBooks/amazonbooks_x0/train_enmf.txt",
          "test_data": "../../data/AmazonBooks/amazonbooks_x0/test_enmf.txt",
          "similarity_measure": "pearson", # searched in [pearson, cosine]
          "num_neighbors": 10, # searched in [5, 10, 20, 50, 100, 150, 200]
          "min_similarity_threshold": 0,
          "renormalize_similarity": False,
          "enable_average_bias": True,
          "metrics": ["Recall(k=20)", "Recall(k=50)", "NDCG(k=20)", "NDCG(k=50)", "HitRate(k=20)", "HitRate(k=50)"]}
model = ItemKNN(params)
model.fit()
model.evaluate()