# Here it should first check what type of file it is. 
#If it is not one of the format, it should return that the format is wrong. 
#import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import unicodedata
import re
import dateutil.parser
import _pickle as cPickle

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct
from tabulate import tabulate
from IPython.display import display, HTML


def import_topic_model(file_name):
    
    root = os.getcwd()
    #root = 'F:/Google Drive/Topic_modelling_analysis/'   # remove this after!!!
    name, extension = os.path.splitext(file_name)
    
    imported_data = cPickle.load(open(str(root) + "/save/" + str(name) + "_saved_topic_model", 'rb'))
    return(imported_data)

   
    
''' Remove the code hereunder later '''

#filelocation = 'F:/Google Drive/Topic_modelling_analysis/save/'
#filelocation = 'C:/Users/tewdewildt/Google Drive/Topic_modelling_analysis/save/'
#file_name = "scopus_nucl_energy.csv"

#with open(str(filelocation) + 'model_and_vectorized_data','rb') as fp:
#    model_and_vectorized_data = pickle.load(fp)
    
#best_number_of_topics = 50
    
#dict_anchor_words = {
#'Safety' : ["safety", "accident"],
#'Value 2' : ["security", "secure", "malicious", "proliferation", "cybersecurity", "cyber", "sabotage", "antisabotage",
#            "terrorism", "theft"],
#'Value 3' : ['sustainability', 'sustainable', 'renewable', 'durability', 'durable'],
#'Value 4' : ["economic viability", "economic", "economic potential", "costs", "cost effective"],
#'Value 5' : ["intergenerational justice", "intergenerational equity", "intergenerational ethics", "intergenerational equality", 
#             "intergenerational relations", "justice", "intergenerational",
#             "future generations", "present generations", "past generations", "waste management", "depleting", "nonrenewable"],
#}

#imported_data = import_topic_model(file_name)


#model_and_vectorized_data = imported_data[0]
#dict_anchor_words = imported_data[1]
#best_number_of_topics = imported_data[2]

#apply_topic_model(model_and_vectorized_data, best_number_of_topics)



#export_topic_model(model_and_vectorized_data, dict_anchor_words, best_number_of_topics, file_name)





#topic_to_evaluate = "Safety"

#topic_in_which_some_keywords_are_found = 36

#random_number_documents_to_return = 5



#df_with_topics = pd.read_pickle(str(filelocation + name) + '_df_with_topics')
#df_with_topics = pd.read_pickle('F:/Google Drive/Topic_modelling_analysis/save/aylien_covid_news_data_GB_df_with_topics')


#print_random_x_articles (df_with_topics, topic_to_evaluate, random_number_documents_to_return)

#list_of_words = ["safety", "safely"]

#print_sample_documents_related_to_topic(df_with_topics, dict_anchor_words, topic_to_evaluate, random_number_documents_to_return)




        

#docs = print_documents_related_to_the_value_that_are_not_yet_in_the_topics(df_with_topics, list_of_words, topic_to_evaluate, topic_in_which_some_keywords_are_found, random_number_documents_to_return)
#print(docs)

#print_sample_documents_related_to_topic_with_keywords(df_with_topics, list_of_words, topic_to_evaluate, random_number_documents_to_return)


#docs = find_documents_related_to_the_value_that_are_not_yet_in_the_topics(df_with_topics, model_and_vectorized_data, dict_anchor_words, list_of_words, topic_to_evaluate)
#print(docs)

#filelocation = 'F:/Google Drive/Topic_modelling_analysis/save/'
#filelocation = 'C:/Users/tewdewildt/Google Drive/Topic_modelling_analysis/save/'

#file_name = "scopus_nucl_energy.csv"
#name, extension = os.path.splitext(file_name)
#file_name = "Covid_data.txt"

#Topic_selected = 0
#df_with_topics = pd.read_pickle(str(filelocation + name) + '_df_with_topics')

#export_documents_related_to_one_topic(df_with_topics, file_name, Topic_selected)


#name, extension = os.path.splitext(file_name)
#max_number_of_documents = 20
#df = pd.read_pickle(filelocation + name)
#print(df.info())

#min_number_of_topics = 10
#max_number_of_topics = 50

#best_number_of_topics = find_best_number_of_topics(df, max_number_of_documents, min_number_of_topics, max_number_of_topics)
#print(best_number_of_topics)

#df_reduced = reduce_df(df, max_number_of_documents)
#make_topic_model(df_reduced)

#number_of_topics = 50
#number_of_documents_in_analysis = 100

#dict_anchor_words = {
#'Value 1' : ["safety", "accident"],
#'Value 2' : ["security", "secure", "malicious", "proliferation", "cybersecurity", "cyber", "sabotage", "antisabotage",
#            "terrorism", "theft"],
#'Value 3' : ['sustainability', 'sustainable', 'renewable', 'durability', 'durable'],
#'Value 4' : ["economic viability", "economic", "economic potential", "costs", "cost effective"],
#'Value 5' : ["intergenerational justice", "intergenerational equity", "intergenerational ethics", "intergenerational equality", 
#             "intergenerational relations", "justice", "intergenerational",
#             "future generations", "present generations", "past generations", "waste management", "depleting", "nonrenewable"],
#}

#list_rejected_words = ["fossil", "coal", "oil", "natural gas", "term", "long term", "short term", "term energy", 
#                       "st century", "st", "century", "decision making", "decision", "making"]

#model_and_vectorized_data = make_anchored_topic_model(df, number_of_topics, number_of_documents_in_analysis, dict_anchor_words, list_rejected_words)
#outcomes = report_topics(model_and_vectorized_data[0], dict_anchor_words)
#df_with_topics = create_df_with_topics(df, model_and_vectorized_data[0], model_and_vectorized_data[1], number_of_topics)
#print(df_with_topics)

#df_with_topics.info()

