import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import _pickle as cPickle
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter1d
from corextopic import corextopic as ct
from datetime import datetime
import dateutil.parser
from dateutil.relativedelta import relativedelta


def topic_int_or_string(Topic_selected, dict_anchor_words):
    
    if type(Topic_selected) == str:
        list_keys = list(dict_anchor_words.keys())
        Topic_selected_number = list_keys.index(Topic_selected)
    else:
        Topic_selected_number = Topic_selected
        
    return Topic_selected_number

def create_vis_values_over_time(df_with_topics, dict_anchor_words, resampling, values_to_include_in_visualisation, smoothing, max_value_y):
    
    copy_df_with_topics = df_with_topics.copy()
    copy_dict_anchor_words = dict_anchor_words.copy()
    
    df_with_topics_freq = copy_df_with_topics.set_index('[Date]').resample(resampling).size().reset_index(name="count")
    df_with_topics_freq = df_with_topics_freq.set_index('[Date]')

    df_frequencies = copy_df_with_topics.set_index('[Date]')
    df_frequencies = df_frequencies.resample(resampling).sum()
       
    list_topics = list(range(len(copy_dict_anchor_words)))
    df_frequencies = df_frequencies[list_topics]
    
    df_frequencies = df_frequencies[list_topics].div(df_with_topics_freq["count"], axis=0)
    combined_df = pd.concat([df_frequencies, df_with_topics_freq], axis=1)
    combined_df = combined_df.fillna(0)
    
    x = pd.Series(combined_df.index.values)
    x = x.dt.to_pydatetime().tolist()

    x = [ z - relativedelta(years=1) for z in x]

    
    name_values = list(copy_dict_anchor_words.keys())
    
    combined_df[list_topics] = combined_df[list_topics] * 100
    combined_df.columns = name_values + ["count"]
       
    if not values_to_include_in_visualisation:
        values_to_include_in_visualisation = name_values

    sigma = (np.log(len(x)) - 1.25) * 1.2 * smoothing

    print(values_to_include_in_visualisation)

    fig, ax1 = plt.subplots()
    for value in values_to_include_in_visualisation:
            ysmoothed = gaussian_filter1d(combined_df[value].tolist(), sigma=sigma)
            ax1.plot(x, ysmoothed, label=str(value), linewidth=2)

    
    ax1.set_xlabel('Time', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Percentage of documents addressing each value \n per unit of time (lines)  (%)', fontsize=12, fontweight="bold")
    ax1.legend(prop={'size': 10})
    
    timestamp_0 = x[0]
    timestamp_1 = x[1]
    

    #width = (time.mktime(timestamp_1.timetuple()) - time.mktime(timestamp_0.timetuple())) / 86400 *.8
    width = (timestamp_1 - timestamp_0).total_seconds() / 86400 * 0.8
       
    ax2 = ax1.twinx()
    ax2.bar(x, combined_df["count"].tolist(), width=width, color='gainsboro')
    ax2.set_ylabel('Number of documents in the dataset \n per unit of time (bars)', fontsize=12, fontweight="bold")
    
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    
    ax1.set_ylim([0,max_value_y])
    

    fig.tight_layout() 
    plt.figure(figsize=(20,14), dpi= 400)
    
    #max_value_y = 100
    
    

    plt.rcParams["figure.figsize"] = [12,6]
    plt.show()
    
def coexistence_values(df_with_topics, dict_anchor_words, resampling, values_selected, smoothing, max_value_y):

    copy_df_with_topics = df_with_topics.copy()
    copy_dict_anchor_words = dict_anchor_words.copy()


    list_columns = copy_df_with_topics.columns.tolist()
    list_topics = list(copy_dict_anchor_words.keys())
    
    index = list_columns.index(0)

    counter = 0
    for i in list_columns:
        if counter >= index and counter < (len(list_topics) + index):
            list_columns[counter]=list_topics[counter - index]
        counter += 1
    
    copy_df_with_topics.columns = list_columns
    
    df_with_topics_freq_value_0 = copy_df_with_topics[[values_selected[0], '[Date]']].set_index('[Date]').resample(resampling).size().reset_index(name="count")
    df_with_topics_freq_value_0 = df_with_topics_freq_value_0.set_index('[Date]')
    
    df_with_topics_selected_topics = copy_df_with_topics[values_selected]
    list_counts = df_with_topics_selected_topics.sum(axis=1).tolist()
    
    counter = 0
    for i in list_counts:
        if i == len(values_selected):
            list_counts[counter] = 1
        else:
            list_counts[counter] = 0
        counter += 1
       
    df_with_topics_sum = copy_df_with_topics[["[Date]"]]
    df_with_topics_sum = df_with_topics_sum.set_index('[Date]')
    
    df_with_topics_sum['all_values_named'] = pd.Series(list_counts, index=df_with_topics_sum.index)
    
    df_with_topics_sum = df_with_topics_sum.resample(resampling).sum()
    
    df_with_topics_selected_topic = df_with_topics_sum.div(df_with_topics_freq_value_0["count"], axis=0)
    df_with_topics_selected_topic = df_with_topics_selected_topic.fillna(0)
    
    x = pd.Series(df_with_topics_selected_topic.index.values)
    x = x.dt.to_pydatetime().tolist()

    df_with_topics_selected_topic = df_with_topics_selected_topic * 100

    sigma = (np.log(len(x)) - 1.25) * 1.2 * smoothing

    fig, ax1 = plt.subplots()
    for word in df_with_topics_selected_topic:
        ysmoothed = gaussian_filter1d(df_with_topics_selected_topic[word].tolist(), sigma=sigma)
        ax1.plot(x, ysmoothed, linewidth=2)
        
        
        ax1.set_xlabel('Time', fontsize=12, fontweight="bold")
    
    
    ax1.set_ylabel('Percentage of articles mentioning \n '+str(values_selected[0])+' also mentioning \n '+str(values_selected[1])+ ' (% of documents)', fontsize=12, fontweight="bold")
    ax1.legend(prop={'size': 8})
    
    ax1.set_ylim([0,max_value_y])
    
    fig.tight_layout() 
    plt.figure(figsize=(20,14), dpi= 400)

    plt.rcParams["figure.figsize"] = [12,6]
    plt.show()
    
    
def inspect_words_over_time(df_with_topics, topic_to_evaluate, list_words, resampling, smoothing, max_value_y):

    df_with_topics_selected_topic = df_with_topics.loc[df_with_topics[topic_to_evaluate] == 1] 
    df_with_topics_selected_topic = df_with_topics_selected_topic.set_index('[Date]')  
    
    df_with_topics_freq = df_with_topics_selected_topic.resample(resampling).size().reset_index(name="count")
    df_with_topics_freq = df_with_topics_freq.set_index('[Date]')
    
    for word in list_words:
        df_with_topics_selected_topic[word] = df_with_topics_selected_topic["[Text_for_analysis]"].str.contains(pat = word).astype(int) #''' Check here '''
    df_with_topics_selected_topic = df_with_topics_selected_topic[list_words] 
    df_with_topics_selected_topic = df_with_topics_selected_topic.resample(resampling).sum()
    
    df_with_topics_selected_topic = df_with_topics_selected_topic.div(df_with_topics_freq["count"], axis=0)
    df_with_topics_selected_topic = df_with_topics_selected_topic.fillna(0)
        
    x = pd.Series(df_with_topics_selected_topic.index.values)
    x = x.dt.to_pydatetime().tolist()
    
    df_with_topics_selected_topic = df_with_topics_selected_topic * 100

    sigma = (np.log(len(x)) - 1.25) * 1.2 * smoothing

    fig, ax1 = plt.subplots()
    for word in df_with_topics_selected_topic:
        ysmoothed = gaussian_filter1d(df_with_topics_selected_topic[word].tolist(), sigma=sigma)
        ax1.plot(x, ysmoothed, label=word, linewidth=2)
    
    ax1.set_xlabel('Time', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Word appearance in documents related to the topic \n over time (% of documents)', fontsize=12, fontweight="bold")
    ax1.legend(prop={'size': 10})
    
    ax1.set_ylim([0,max_value_y])
    
    fig.tight_layout() 
    plt.figure(figsize=(20,14), dpi= 400)

    plt.rcParams["figure.figsize"] = [12,6]
    plt.show()

def inspect_words_over_time_based_on_most_frequent_words(df_with_topics, dict_anchor_words, model_and_vectorized_data, topic_to_evaluate, number_of_words, resampling, smoothing, max_value_y):
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
    list_words = list(list(zip(*model_and_vectorized_data[0].get_topics(topic=topic_to_evaluate_number, n_words=number_of_words)))[0])
    inspect_words_over_time(df_with_topics, topic_to_evaluate_number, list_words, resampling, smoothing, max_value_y)

def inspect_words_over_time_based_on_own_list(df_with_topics, dict_anchor_words, topic_to_evaluate, list_words, resampling, smoothing, max_value_y):
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
    inspect_words_over_time(df_with_topics, topic_to_evaluate_number, list_words, resampling, smoothing, max_value_y)

''' Remove the code hereunder later '''
     
    
#filelocation = 'F:/Google Drive/Topic_modelling_analysis/save/'
#file_name = "scopus_nucl_energy.csv"
#name, extension = os.path.splitext(file_name)

#df_with_topics = pd.read_pickle(str(filelocation + name) + '_df_with_topics')

#df_with_topics = pd.read_pickle('F:/Google Drive/Topic_modelling_analysis/save/aylien_covid_news_data_GB_df_with_topics')
#df_with_topics = pd.read_pickle('C:/Users/tewdewildt/Google Drive/Topic_modelling_analysis/save/scopus_nucl_energy_df_with_topics')
#df_with_topics = pd.read_pickle('C:/Users/tewdewildt/Google Drive/Topic_modelling_analysis/save/aylien_covid_news_data_all_df_with_topics')




#df_with_topics =  pd.read_pickle("../save/df_with_topics")


#resampling = "Y"

#dict_anchor_words = {
#"Sustainability" : ["sustainability", "sustainable", "renewable", 'durability', 'durable'],        
#"Economic viability" : ["economic viability", "economic", "economic potential", "costs", "cost effective", "cost"],
#"Affordability" : ["affordability", "affordable", "energy security", "low cost", "income", "poor", 
#                   "poverty", "low income", "accessibility"],
#"Availability" : ["availability", "reliability", "reliable", "security of supply"],
#"Safety & Health" : ["safety", "health", "accident", "accidents"],
#"Justice" : ["justice", "fairness", "social equity", "injustice", "injustices", "equity",
#            "social fairness", "inequality", "inequalities",],
#"Privacy & security" : ["privacy", "privacy concerns", "privacy preserving", "data privacy", 
#             "privacy perservation", "cyber"],
#
#}


#values_selected = ['Health & Safety', 'Privacy']


#coexistence_values(df_with_topics, dict_anchor_words, resampling, values_selected)

#values_to_include_in_visualisation = []
#smoothing = 1
#max_value_y = 60

#print(df_with_topics)

#df_with_topics = df_with_topics[df_with_topics['[Date]'] >= dateutil.parser.parse(str(1980))]

#create_vis_values_over_time(df_with_topics, dict_anchor_words, resampling, values_to_include_in_visualisation, smoothing, max_value_y)    





#filelocation = 'F:/Google Drive/Topic_modelling_analysis/save/'
#filelocation = 'C:/Users/tewdewildt/Google Drive/Topic_modelling_analysis/save/'
#file_name = "scopus_nucl_energy.csv"
#name, extension = os.path.splitext(file_name)
#df_with_topics = pd.read_pickle(str(filelocation + name) + '_df_with_topics')

#import pickle
#with open(str(filelocation)+"model_"+str(name),'rb') as fp:
#    model_and_vectorized_data = pickle.load(fp)

    
#imported_data = pickle.load(open( "C:/Users/tewdewildt/Google Drive/Topic_modelling_analysis/save/aylien_covid_news_data_all_saved_topic_model", 'rb'))

#resampling = 'D'

#topic_to_evaluate = 'Privacy'
#number_of_words = 10

#list_words = ["peace", "safety", "sustainability", "the safety"]

#print(model_and_vectorized_data)
#inspect_words_over_time_based_on_most_frequent_words(df_with_topics, imported_data[1], imported_data[0], topic_to_evaluate, number_of_words, resampling)

#inspect_words_over_time_based_on_most_frequent_words(df_with_topics, model_and_vectorized_data, topic_to_evaluate, number_of_words, resampling)




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


#create_vis_values_over_time(df_with_topics, dict_anchor_words)












#filelocation = 'F:/Google Drive/Topic_modelling_analysis/save/'
#filelocation = 'C:/Users/tewdewildt/Google Drive/Topic_modelling_analysis/save/'
#file_name = "scopus_nucl_energy.csv"
#file_name = "Covid_data.txt"

#name, extension = os.path.splitext(file_name)
#max_number_of_documents = 100
#df = pd.read_pickle(filelocation + name)
#print(df.info())

#min_number_of_topics = 50
#max_number_of_topics = 250

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

#df_with_topics = pd.read_pickle("./dummy.pkl")




