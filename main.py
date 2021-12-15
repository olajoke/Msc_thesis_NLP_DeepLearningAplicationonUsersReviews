import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer
import re

from change import fast_clustering,tensor_2d_dimension_reducer,create_graph,silhouette_clustering,kmeans_clustering
"""## Loading data"""

def get_count(string):
  count = 0
  for x in string:
    if x.isalpha():
      count = count+1
  if count >10:
    return True
  return False

my_file = open("/changeadvisor_Code/reviews/org.geometerplus.zlibrary.ui.android.txt", "r")
content = my_file.read()
content_list = content.split("\n")
my_file.close()
#Only keeping English alphabets
content_list = [re.sub(r'[^A-Za-z0-9 .,?]+', "" ,i) for i in content_list if get_count(i)]
#removing empty strings
content_list=[i.strip() for i in content_list]
content_list = list(filter(None, content_list))

content_list= content_list[:]# Specify Number of rows to be used ---->change later

# loading the bert model
bert_model = SentenceTransformer('bert-base-uncased')


#---------------------------Visualization Fast Clustering------------------------------

input_list0=content_list[:]

input_fast_data,embedding_fast_model= fast_clustering(bert_model,input_list0,content_list)##Enter Model name here and run
ref_dframe=input_fast_data['text'].values
clusters_number=  50
graph_embeddings= tensor_2d_dimension_reducer(embedding_fast_model)

create_graph(graph_embeddings,input_fast_data,'Fast_Clustering',clusters_number)


#-----------------------------Visualization Silhouette---------------------

input_list0=content_list[:]
input_list0=list(set(input_list0).intersection(ref_dframe))
input_silhouette_data,embedding_silhouette_model= silhouette_clustering(bert_model,input_list0,5,30)##Enter Model name here and run

clusters_number=  50
graph_embeddings= tensor_2d_dimension_reducer(embedding_silhouette_model)

create_graph(graph_embeddings,input_silhouette_data,'Silhouette',clusters_number)

#---------------------------------Visualization Kmeans Clustering(elbow method)------------------------------

input_list0=content_list[:]
input_list0=list(set(input_list0).intersection(ref_dframe))

input_kmeans_data,embedding_kmeans_model= kmeans_clustering(bert_model,input_list0,5,30)##Enter Model name here and run
clusters_number=  50

graph_embeddings= tensor_2d_dimension_reducer(embedding_kmeans_model)

create_graph(graph_embeddings,input_kmeans_data,'K_Means',clusters_number)





