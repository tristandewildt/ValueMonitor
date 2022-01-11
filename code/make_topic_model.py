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
#from tabulate import tabulate
from IPython.display import display, HTML


''' First select a number of documents from the data set to create the topic model more rapidly'''

def topic_int_or_string(Topic_selected, dict_anchor_words):
    
    if type(Topic_selected) == str:
        list_keys = list(dict_anchor_words.keys())
        Topic_selected_number = list_keys.index(Topic_selected)
    else:
        Topic_selected_number = Topic_selected
        
    return Topic_selected_number

def reduce_df(df, max_number_of_documents):

    ''' Here we should include an option in case the dataset is not well spread over time '''
    df_reduced = df.sample(min(max_number_of_documents, len(df)))
    return df_reduced

def vectorize(df):
    
    vectorizer = TfidfVectorizer(
        max_df=.5,
        min_df=10,
        max_features=None,
        ngram_range=(1, 2),
        norm=None,
        binary=True,
        use_idf=False,
        sublinear_tf=False
    )

    vectorizer = vectorizer.fit(df['[Text_for_analysis]'])
    tfidf = vectorizer.transform(df['[Text_for_analysis]'])
    vocab = vectorizer.get_feature_names()
    vectorized_data = [vectorizer, tfidf, vocab]
    
    return vectorized_data

def make_topic_model(df, number_of_topics, anchors):
    
    vectorized_data = vectorize(df)
    
#    vectorizer = vectorized_data[0]
    tfidf = vectorized_data[1]
    vocab = vectorized_data[2]

    model = ct.Corex(n_hidden=number_of_topics, seed=42)
    model = model.fit(
            tfidf,
            words=vocab,
            anchors=anchors,
            anchor_strength=3) # Check whether this still works when there are no anchor words   

    return model

def find_best_number_of_topics(df, number_of_documents_in_analysis, min_number_of_topics, max_number_of_topics):
    ''' Think here what could be some errors that people could make with regard to input data '''
    
    df_reduced = reduce_df(df, number_of_documents_in_analysis)
    
    interval = (max_number_of_topics - min_number_of_topics) / 4
    list_topics_to_try = np.arange(min_number_of_topics, (max_number_of_topics + interval), interval).tolist()
    list_topics_to_try = [int(i) for i in list_topics_to_try]
    
    dict_topic_correlation = {}
    dict_topic_models = {}
    
    for number_of_topics in list_topics_to_try:
        print("Working on model with "+str(number_of_topics)+" topics...") # Might need to put this as info
        
        anchors = []
        model = make_topic_model(df_reduced, number_of_topics, anchors)
        
        dict_topic_correlation[number_of_topics] = np.sum(model.tcs)
        dict_topic_models[number_of_topics] = model
        
    fig, ax1 = plt.subplots()
    
    ax1.plot(list(dict_topic_correlation.keys()),list(dict_topic_correlation.values()))
    ax1.set_xlabel('Number of topics', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Total correlation', fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()

    best_number_of_topics = max(dict_topic_correlation, key=dict_topic_correlation.get)
    
    return best_number_of_topics

def make_anchored_topic_model(df, number_of_topics, number_of_documents_in_analysis, dict_anchor_words, list_anchor_words_other_topics, list_rejected_words):
    ''' Think here what could be some errors that people could make with regard to input data '''
    
    df_reduced = reduce_df(df, number_of_documents_in_analysis)   
    vectorized_data = vectorize(df_reduced)
    vocab = vectorized_data[2]
    
    anchors = [[]] * number_of_topics
    
    counter = 0
    for key, value in dict_anchor_words.items():
        value_lowercase = value
        for i in range(len(value_lowercase)):
            value_lowercase[i] = value_lowercase[i].lower()
        anchors[counter] = value_lowercase
        counter += 1
    for i in list_anchor_words_other_topics:
        anchors[counter]=i
        counter += 1
    anchors[counter]=list_rejected_words
    
    anchors = [
        [a for a in topic if a in vocab]
        for topic in anchors]
    
    model = make_topic_model(df_reduced, number_of_topics, anchors)
    model_and_vectorized_data = [model, vectorized_data]
    
    return model_and_vectorized_data

def report_topics(model, dict_anchor_words, number_of_words_per_topic):
        
    list_values = []
    for key, value in dict_anchor_words.items(): 
        list_values.append(key)

    for i, topic_ngrams in enumerate(model.get_topics(n_words=number_of_words_per_topic)):
        topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
        
        if len(list_values) > i:
            print("Topic #{} ({}): {}".format(i, list_values[i], ", ".join(topic_ngrams)))
        else:
            print("Topic #{}: {}".format(i, ", ".join(topic_ngrams)))

def create_df_with_topics(df, model, vectorized_data, best_number_of_topics):
    vectorizer = vectorized_data[0]
    
    tfidf = vectorizer.transform(df['[Text_for_analysis]']) 
    
    df_documents_topics = pd.DataFrame(
        model.transform(tfidf), 
        columns=[i for i in range(best_number_of_topics)]
    ).astype(float)

    df_documents_topics.index = df.index
    df_with_topics = pd.concat([df, df_documents_topics], axis=1)
      
    return df_with_topics

def export_documents_related_to_one_topic(df_with_topics, dict_anchor_words, file_name, Topic_selected):
    
    Topic_selected_number = topic_int_or_string(Topic_selected, dict_anchor_words)
    
    root = '/gdrive/My Drive/Topic_modelling_analysis/'
    name, extension = os.path.splitext(file_name)
    
    df_selected = df_with_topics[df_with_topics[Topic_selected_number] == 1] 
    df_selected = DataFrame(df_selected,columns=["[Text_for_analysis]", "[Date]"])

    df_selected.to_csv(str(root) + "save/" + str(name) + "_topic_"+str(Topic_selected)+".csv", index = False)
    
def export_topic_model(model_and_vectorized_data, dict_anchor_words, best_number_of_topics, file_name):
    
    root = os.getcwd()
    name, extension = os.path.splitext(file_name)
    
    saved_data = [model_and_vectorized_data, dict_anchor_words, best_number_of_topics]
    
    cPickle.dump(saved_data, open(str(root) + "/save/" + str(name) + "_saved_topic_model", 'wb'))

    
def find_documents_related_to_the_value_that_are_not_yet_in_the_topics(df_with_topics, model_and_vectorized_data, dict_anchor_words, list_of_words, topic_to_evaluate, number_of_words_per_topic_to_show):
        
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
       
    listToStr = ', '.join([str(elem) for elem in list_of_words])
    name_column_counts = "Number of documents found in each topic with keywords '" +str(listToStr)+"' which have not been assigned to topic "+str(topic_to_evaluate)+"."

    df_selected = df_with_topics[df_with_topics["[Text_for_analysis]"].str.contains('|'.join(list_of_words))]
    df_selected = df_selected[df_selected[topic_to_evaluate_number] == 0]
    
    print(str(len(df_selected))+" documents found that contains words in the list and have not been attributed to the topic of interest.")
       
    model = model_and_vectorized_data[0]
    list_topics = list(range(len(model.get_topics())))
    list_topics.remove(topic_to_evaluate_number)
    df_column_topics = df_selected[list_topics]
    
    df_column_topics = pd.DataFrame(df_column_topics.sum(axis=0))
    df_column_topics = df_column_topics.rename(columns={0: name_column_counts})

    list_values = []
    for key, value in dict_anchor_words.items(): 
        list_values.append(key)

    words_topics = []
    for i, topic_ngrams in enumerate(model.get_topics(n_words=number_of_words_per_topic_to_show)):
        if i in list_topics:
            topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
            if len(list_values) > i:
                words_topics.append("Topic #{} ({}): {}".format(i, list_values[i], ", ".join(topic_ngrams)))
            else:
                words_topics.append("Topic #{}: {}".format(i, ", ".join(topic_ngrams)))

    df_column_topics.insert(0, "Topics", words_topics)
    df_column_topics = df_column_topics.sort_values(by=[name_column_counts], ascending=False)

    display(HTML(df_column_topics.to_html(justify = "center")))
    
def sample_documents(df_selected, random_number_documents_to_return, text_table):

    df_selected_texts = pd.DataFrame(df_selected["[Text_for_analysis]"].sample(n = min(random_number_documents_to_return, len(df_selected))))
    
    df_selected_texts["[Text_for_analysis]"] = df_selected_texts["[Text_for_analysis]"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    df_selected_texts = df_selected_texts.rename(columns={"[Text_for_analysis]": text_table})

    display(HTML(df_selected_texts.to_html(justify = "center")))

def print_documents_related_to_the_value_that_are_not_yet_in_the_topics(df_with_topics, dict_anchor_words, list_of_words, topic_to_evaluate, topic_in_which_some_keywords_are_found, random_number_documents_to_return):
    
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
    
    df_selected = df_with_topics[df_with_topics["[Text_for_analysis]"].str.contains('|'.join(list_of_words))]
    df_selected = df_selected[(df_selected[topic_to_evaluate_number] == 0) & (df_selected[topic_in_which_some_keywords_are_found] == 1)]
    
    listToStr = ', '.join([str(elem) for elem in list_of_words])
    text_table = "Random " + str(random_number_documents_to_return) + " documents in topic " + str(topic_in_which_some_keywords_are_found) + " with keywords '" + str(listToStr) + "' that have not been assigned to topic " + str(topic_to_evaluate) + "."
    
    sample_documents(df_selected, random_number_documents_to_return, text_table)
    
def print_sample_documents_related_to_topic(df_with_topics, dict_anchor_words, topic_to_evaluate, random_number_documents_to_return):
    
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)

    df_selected = df_with_topics[df_with_topics[topic_to_evaluate_number] == 1]
    text_table = "Random " + str(random_number_documents_to_return) + " documents in topic " + str(topic_to_evaluate) + "."
    
    sample_documents(df_selected, random_number_documents_to_return, text_table)

def print_sample_documents_related_to_topic_with_keywords(df_with_topics, dict_anchor_words, list_of_words, topic_to_evaluate, random_number_documents_to_return):
    
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
    
    df_selected = df_with_topics[df_with_topics["[Text_for_analysis]"].str.contains('|'.join(list_of_words))]
    df_selected = df_selected[df_selected[topic_to_evaluate_number] == 1]
    
    listToStr = ', '.join([str(elem) for elem in list_of_words])
    text_table = "Random " + str(random_number_documents_to_return) + " documents in topic " + str(topic_to_evaluate) + " with keywords '" + str(listToStr) + "."
    
    sample_documents(df_selected, random_number_documents_to_return, text_table)
    

   
    
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

