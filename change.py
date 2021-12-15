import pandas as pd
import numpy as np
import os
from kneed import KneeLocator
import torch
import json 
from summarizer.sbert import SBertSummarizer
import random
from random import randrange
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.cluster import KMeans #K-Means Clustering
import plotly as py
import plotly.graph_objs as go
from plotly.offline import  plot, iplot
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = stopwords.words('english') #this depends on each language
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
import re


#------------------------Kmeans with elbow ------------------------------------------------------------


def get_k(embeddings,starting_range = 4,ending_range=15):
  
  k_range = range(starting_range,ending_range)
  inertias = []
  for k in k_range:
      km_cluster = KMeans(n_clusters = k,algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
      n_init=10, random_state=0, tol=0.0001, verbose=0)
      km_cluster.fit(embeddings)
      inertias.append(km_cluster.inertia_)
  
  elbow_locator = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
  elbow_point = elbow_locator.knee - 1
  return elbow_point

def cluster_reviews(embeddings,text,best_k):
  clustering_model = KMeans(n_clusters=best_k,algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
      n_init=10, random_state=0, tol=0.0001, verbose=0)
  clustering_model.fit(embeddings)
  cluster_assignment = clustering_model.labels_
  data = pd.DataFrame(list(zip(text, cluster_assignment)),
               columns =['text', 'target'])
  data.to_csv('Clustered_Data.csv')

def kmeans_clustering(bert_model, sentence_list,start,end):
  k_means_embedding= bert_model.encode(sentence_list)
  k_value = get_k(k_means_embedding,start,end)
  cluster_reviews(k_means_embedding,sentence_list,k_value)
  k_means_dataframe=pd.read_csv('Clustered_Data.csv')
  k_means_dataframe=k_means_dataframe.iloc[:,1:]
  k_means_dataframe.columns =['text', 'target']

  return k_means_dataframe,k_means_embedding



#---------------------------------- Silhouette_Method - -----------------------------------------

def silhouette_clustering(bert_model,content_list,start=2,end=20):

  corpus_embeddings= bert_model.encode(content_list)

  k_range_silhoutte = range(start,end)
  silhoutte_score = [] # within-cluster-sum of Squared 
  for k in k_range_silhoutte:
    km = KMeans(n_clusters=k,algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                    n_init=10,random_state=0, tol=0.0001, 
                    verbose=0).fit(corpus_embeddings).labels_
    score = metrics.silhouette_score(corpus_embeddings,km, metric = "euclidean",random_state=0)
    silhoutte_score.append(score)

  df = pd.DataFrame({"Clusters": k_range_silhoutte, "silhoutte_score": silhoutte_score})

  kn = KneeLocator(df.Clusters, df.silhoutte_score, curve='convex', direction='decreasing')
  S_elbow_point = kn.knee
  S_elbow_point


  clustering_model = KMeans(n_clusters=S_elbow_point,algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                            n_init=10, random_state=0, tol=0.0001, verbose=0)
  clustering_model.fit(corpus_embeddings)
  cluster_assignment = clustering_model.labels_
  silhouette_data = pd.DataFrame(list(zip(content_list, cluster_assignment)), columns =['text', 'target'])

  return silhouette_data, corpus_embeddings


#--------------------------------------- Fast Clustering -------------------------------

def cluster_detail(clusters):
  clus=[]
  for i in range(len(clusters)):

    clus.append(len(clusters[i]))
  print('Detected Clusters: '+str(len(clusters)))
  print("Matched text in each Clusters: "+str(clus))
  print("Total Sentences in all clusters: "+str(sum(clus)))
  
def flat_cluster(clusters):
  return [item for sublist in clusters for item in sublist]



def fast_clustering(model_fast,sentences_list,content_list):

  corpus_sentences = sentences_list ########## Change No. of Rows here --->TODO: remove later and replace encoded uncased BERT Version
  print("Encoding the corpus...") ####
  fast_embeddings0 = model_fast.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
  clusters = util.community_detection(fast_embeddings0, min_community_size=1, threshold=0.75)
  a=flat_cluster(clusters)
  a2 = [i for i, element in enumerate(content_list) if i not in a]
  unidentified_clusters=[[i] for i in a2]
  clusters_final=clusters+unidentified_clusters
  cluster_min_frequency=5 #########Set Freqency of Clusters element here
  cluster_list_final=[]
  for i in clusters_final:
    if len(i) >= cluster_min_frequency:
      cluster_list_final.append(i)

  fast_cluster_indexes1= flat_cluster(cluster_list_final)
  fast_cluster2= [element for i, element in enumerate(content_list) if i in fast_cluster_indexes1]

  count_cluster=0
  fast_cluster_dataframe = pd.DataFrame(columns =['text', 'target'])
  for index in cluster_list_final:
    text= [element for i, element in enumerate(content_list) if i in index]
    text_index=[int(i) for i, element in enumerate(content_list) if i in index]
    cluster=[count_cluster]*len(text)
    fast_cluster_data2 = pd.DataFrame(columns =['text', 'target'])
    fast_cluster_data2['text']=pd.Series(text)
    fast_cluster_data2['target']=pd.Series(cluster)
    fast_cluster_dataframe=pd.concat([fast_cluster_dataframe, fast_cluster_data2],ignore_index=True)
    count_cluster+=1

  fast_embeddings2= model_fast.encode(list(fast_cluster_dataframe['text']), batch_size=64, show_progress_bar=True, convert_to_tensor=True)
  return fast_cluster_dataframe, fast_embeddings2.cpu()


#-------------Extractive Summarization---------------------------

def extractive_summarization(input_dataframe):
  input_dataframe=input_dataframe.sample(frac=1)
  text= input_dataframe['text'].values.tolist()
  text=[str(i) for i in text]
  text='.'.join(text)

  model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
  output = model(text, num_sentences=3)
  return output


#---------------------------LDA_Model-----------------------------------

def lda_model_2(input_dataframe):

  def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

  data=input_dataframe['text'].values.tolist()
  data_words = list(sent_to_words(data))
  bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
  trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

  bigram_mod = gensim.models.phrases.Phraser(bigram)
  trigram_mod = gensim.models.phrases.Phraser(trigram)

  def remove_stopwords(texts):
      return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

  def make_bigrams(texts):
      return [bigram_mod[doc] for doc in texts]

  def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
      texts_out = []
      for sent in texts:
          doc = nlp(" ".join(sent)) 
          texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
      return texts_out

  data_words_nostops = remove_stopwords(data_words)
  data_words_bigrams = make_bigrams(data_words_nostops)
  nlp = spacy.load('en', disable=['parser', 'ner'])
  data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
  id2word = corpora.Dictionary(data_lemmatized)
  texts = data_lemmatized
  corpus = [id2word.doc2bow(text) for text in texts]

  lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=10, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=4,
                                            alpha='auto',
                                            per_word_topics=True)
  topic_number_dict={}
  for idx, topic in lda_model.print_topics(-1):
      topic_number_dict[idx]= topic
  output = re.sub('[^A-Za-z]+', ' ', topic_number_dict[1]).strip()
  output2=' '.join(output.split(' ')[:7])
  return np.char.capitalize(output2)



def tensor_2d_dimension_reducer(embeddings):
  pca =PCA(n_components=2)
  graph_2d_embeddings = pca.fit_transform(embeddings)

  return graph_2d_embeddings
# function to create a graph
def create_graph(data_tensor_2d,input_dataframe1,method_name, c_number=10):
  '''
  data_tensor_2d: Clustering model Embedding that are reduced to 2_Dimension from higher dimensions

  input_dataframe: Dataframe containing 'text' and 'target' columns after clustering

  c_number: Maximum number of cluster, we want to view (As Fast Clustering model can generate several hundered clusters)
  '''
  input_dataframe=input_dataframe1.copy()
  X1=data_tensor_2d

  xc1=[]
  xc2=[]
  for i in X1:
    xc1.append(i[0])
    xc2.append(i[1])
    
  input_dataframe['x_1_component']=xc1
  input_dataframe['x_2_component']=xc2
  input_dataframe['LDA_Keywords']=np.nan
  input_dataframe['Bert_Summary']=np.nan
  input_dataframe['Frequency']=np.nan

  data_e=[]

  for c_no in range(c_number):
    data_set0=input_dataframe[input_dataframe['target']==c_no]
    frequency=int(len(data_set0))
    try:
      topic=lda_model_2(data_set0)
      summary=extractive_summarization(data_set0)
    except:
      try:
        # topic=extractive_summarization(data_set0)
        # summary=topic
        topic='No Cluster'
        Summary='No Summary'
      except:
        continue
    input_dataframe.loc[input_dataframe['target']==c_no ,'LDA_Keywords']=topic
    input_dataframe.loc[input_dataframe['target']==c_no ,'Bert_Summary']=summary
    input_dataframe.loc[input_dataframe['target']==c_no ,'Frequency']=frequency
    topic = re.sub(" ",", ",str(topic))
    #Instructions for building the 2-D plot
    #trace1 is for 'Clusters'
    trace = go.Scatter(
                        x = data_set0["x_1_component"],
                        y = data_set0["x_2_component"],
                        mode = "markers",
                        name = str(c_no)+': '+str(topic),
                        marker = dict(color = 'rgba('+str(randrange(255))+','+ str(randrange(255))+','+ str(randrange(255))+','+str(0.8)+')'),
                        text = None)
    data_e.append(trace)

  #input_dataframe.drop(['x_1_component','x_2_component'])
  input_dataframe=input_dataframe.drop(['x_1_component','x_2_component'],axis=1)
  input_dataframe= input_dataframe.sample(frac = 1)
  input_dataframe.to_csv(method_name +'_Output_dataframe.csv')

  title = "Visualizing " +method_name+ " Clusters in Two Dimensions Using PCA"

  layout = dict(title = title,
                xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False),
                width=1400, height=900
              )

  fig=go.Figure(data = data_e, layout = layout)
  fig.show()
  fig.write_image(method_name+'.png')

