import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import _pickle as cPickle
import pickle
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter1d
from corextopic import corextopic as ct
from datetime import datetime
import dateutil.parser
from dateutil.relativedelta import relativedelta
from matplotlib.ticker import FormatStrFormatter
from colorama import Fore, Back, Style
from umap import UMAP
from typing import List
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment import SentimentIntensityAnalyzer


def top_topics_on_values(df_with_topics, selected_value, dict_anchor_words, topics_weights, topics_to_remove_int, top_topics_to_show):
    
    dict_values = {}
    counter = 0
    for value, words in dict_anchor_words.items():
        dict_values[value]=counter
        counter = counter + 1
    
    df_to_evaluate = df_with_topics
    df_to_evaluate = df_to_evaluate.loc[(df_to_evaluate[dict_values[selected_value]] == 1)]
    
    list_columns_on_topics = list(range(len(dict_anchor_words), len(topics_weights), 1))
    list_empty_topics = []
    for topic, words in topics_weights.items():
        if len(words)==0:
            topic_int = topic[topic.find('#')+len('#'):topic.rfind(')')]
            list_empty_topics.append(int(topic_int))
    list_columns_on_topics = [x for x in list_columns_on_topics if x not in list_empty_topics]
    list_columns_on_topics = [x for x in list_columns_on_topics if x not in topics_to_remove_int]
    df_to_evaluate = df_to_evaluate[list_columns_on_topics]
    
    count_df_to_evaluate = df_to_evaluate.sum()
    count_df_to_evaluate = count_df_to_evaluate.sort_values(ascending=False)
    count_df_to_evaluate = count_df_to_evaluate[:top_topics_to_show]
    
    dict_dataset_short = count_df_to_evaluate.to_dict()
    dict_dataset_short_names = {}
    for topic_int, count in dict_dataset_short.items():
        words_weights = topics_weights["Topic #"+str(topic_int)+"#"]
        list_words_weights = "Topic "+str(topic_int)+": "+", ".join(list(words_weights.keys())[:5])
        dict_dataset_short_names[list_words_weights]=count
    
    plt.barh(list(dict_dataset_short_names.keys()), list(dict_dataset_short_names.values()))
    plt.gca().invert_yaxis()
        
    plt.rcParams.update({'font.size': 16})
    plt.title('Top topics associated with the value '+str(selected_value))
    plt.xlabel('counts')
    plt.show()



def top_topics_on_values_over_time(df_with_topics, selected_value, dict_anchor_words, topics_weights, top_topics_to_show, topics_to_remove_int, smoothing, max_value_y, resampling):
  
    df_to_evaluate = df_with_topics
    
    dict_values = {}
    counter = 0
    for value, words in dict_anchor_words.items():
        dict_values[value]=counter
        counter = counter + 1
        
    df_to_evaluate = df_to_evaluate.loc[(df_to_evaluate[dict_values[selected_value]] == 1)]

    df_to_evaluate = df_to_evaluate.set_index('date')
    
    df_with_topics_freq = df_to_evaluate.resample(resampling).size().reset_index(name="count")
    df_with_topics_freq = df_with_topics_freq.set_index('date')
        
    df_to_evaluate = df_to_evaluate.fillna("")

    number_of_topics = len(topics_weights)
    list_int_topics = list(range(number_of_topics))
    list_int_values = list(range(len(dict_anchor_words)))
    list_int_topics = [topic for topic in list_int_topics if topic not in list_int_values]
    df_to_evaluate = df_to_evaluate[list_int_topics]

    topic_names = {}
    empty_topics = []
    for index_topic in range(len(list_int_topics)):
        topic_int = index_topic + len(dict_anchor_words)
        words_weights = topics_weights["Topic #"+str(topic_int)+"#"]
        if len(words_weights) > 0:
            list_words_weights = "Topic "+str(topic_int)+": "+", ".join(list(words_weights.keys())[:5])
            topic_names[topic_int]=list_words_weights
        else: 
            empty_topics.append(topic_int)

    df_to_evaluate = df_to_evaluate.rename(columns=topic_names)
    topics_to_remove_str = empty_topics
    for i in topics_to_remove_int:
        topics_to_remove_str.append(topic_names[i])
    df_to_evaluate = df_to_evaluate.drop(columns=topics_to_remove_str)

    df_to_evaluate = df_to_evaluate.resample(resampling).sum()
    count_df_to_evaluate = df_to_evaluate.sum()

    count_df_to_evaluate = count_df_to_evaluate.sort_values(ascending=False)
    count_df_to_evaluate = count_df_to_evaluate[:top_topics_to_show]

    percentage_df_to_evaluate = count_df_to_evaluate.divide(count_df_to_evaluate.sum(), fill_value=0)
    percentage_df_to_evaluate = percentage_df_to_evaluate * 100
    list_topics_above_threshold = list(count_df_to_evaluate.index.values)

    df_to_evaluate = df_to_evaluate[list_topics_above_threshold]
      
    df_to_evaluate = df_to_evaluate.div(df_with_topics_freq["count"], axis=0)
    df_to_evaluate = df_to_evaluate.fillna(0)
        
    x = pd.Series(df_to_evaluate.index.values)
    x = x.dt.to_pydatetime().tolist()
      
    x = [ z - relativedelta(years=1) for z in x]
        
    df_to_evaluate = df_to_evaluate * 100

      
    sigma = (np.log(len(x)) - 1.25) * 1.2 * smoothing
          
    counter = 0
    fig, ax1 = plt.subplots()
    for word in df_to_evaluate:
        ysmoothed = gaussian_filter1d(df_to_evaluate[word].tolist(), sigma=sigma)
        ax1.plot(x, ysmoothed, label=word, linewidth=2)
        counter = counter + 1
          
    ax1.set_xlabel('Time', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Percentage of articles', fontsize=12, fontweight="bold")
    ax1.legend(prop={'size': 10})
      
    timestamp_0 = x[0]
    timestamp_1 = x[1]
      

    #width = (time.mktime(timestamp_1.timetuple()) - time.mktime(timestamp_0.timetuple())) / 86400 *.8
    width = (timestamp_1 - timestamp_0).total_seconds() / 86400 * 0.8
      
    df_to_evaluate["count"]=df_with_topics_freq["count"]
      
    ax2 = ax1.twinx()
    ax2.bar(x, df_to_evaluate["count"].tolist(), width=width, color='gainsboro')
    ax2.set_ylabel('Number of documents in the selected dataset (bars)', fontsize=12, fontweight="bold")
      
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)

          
    ax1.set_ylim([0,max_value_y])
    ax1.legend(prop={'size': 10})
      
    plt.rcParams["figure.figsize"] = [12,6]
    plt.title("Top "+str(top_topics_to_show)+" topics discussed in relation to the value "+str(selected_value), fontsize=14, fontweight="bold")
    plt.show()


def values_in_different_datasets(df_with_topics, dict_anchor_words):
    
    list_values = list(dict_anchor_words.keys())
    list_values_int = []
    for i in list_values:
        list_values_int.append(list(dict_anchor_words.keys()).index(i))
        
    list_datasets = df_with_topics['dataset'].unique().tolist()
        
    series_perc = {}
    
    for dataset in list_datasets:
        df_with_topics_dataset = df_with_topics[df_with_topics['dataset'] == dataset]
        df_with_topics_sum_dataset = df_with_topics_dataset[[c for c in df_with_topics_dataset.columns if c in list_values_int]]
        df_with_topics_sum_dataset.columns = list_values
        df_sum = df_with_topics_sum_dataset.sum(numeric_only=True)
        series_perc[dataset] = df_sum.apply(lambda x: x / len(df_with_topics_sum_dataset) * 100)
    
    #print(series_perc)
    
    df_perc = {k:v.to_frame() for k, v in series_perc.items()}
    df_perc_all = df_perc[list_datasets[0]]

    counter = 0
    for dataset in list_datasets:
        if counter > 0:
            df_perc_all = pd.concat([df_perc_all, df_perc[dataset].reindex(df_perc_all.index)], axis=1)
        counter = counter + 1
    
    df_perc_all.columns = list_datasets
    
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    
    c = {}
    counter = 0
    for dataset in list_datasets:
        c[dataset] = colors[counter]
        counter = counter + 1
        if counter >= len(colors):
            counter = 0

    plt.rcParams.update({'font.size': 15})
    ax = df_perc_all.plot(kind='bar', figsize=(15,10), color=c, width = 0.75)
    ax.set_ylabel("%")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.title("Values in different datasets")
    
    for dataset in list_datasets:
        print("Number articles in "+str(dataset)+": "+str(len(df_with_topics[df_with_topics['dataset'] == dataset])))
    
def value_in_different_datasets(df_with_topics, dict_anchor_words, selected_value):
    
    list_values = list(dict_anchor_words.keys())
    list_values_int = []
    for i in list_values:
        list_values_int.append(list(dict_anchor_words.keys()).index(i))
    
    value_list = [selected_value]
    value_int_list = [list_values_int[list_values.index(selected_value)]]
    
    list_datasets = df_with_topics['dataset'].unique().tolist()

    
    series_perc = {}
    
    for dataset in list_datasets:
        df_with_topics_dataset = df_with_topics[df_with_topics['dataset'] == dataset]
        df_with_topics_sum_dataset = df_with_topics_dataset[[c for c in df_with_topics_dataset.columns if c in value_int_list]]
        df_with_topics_sum_dataset.columns = value_list
        df_sum = df_with_topics_sum_dataset.sum(numeric_only=True)
        series_perc[dataset] = df_sum.apply(lambda x: x / len(df_with_topics_sum_dataset) * 100)
    
    #print(series_perc)
    
    df_perc = {k:v.to_frame() for k, v in series_perc.items()}
    df_perc_all = df_perc[list_datasets[0]]

    counter = 0
    for dataset in list_datasets:
        if counter > 0:
            df_perc_all = pd.concat([df_perc_all, df_perc[dataset].reindex(df_perc_all.index)], axis=1)
        counter = counter + 1
    
    df_perc_all.columns = list_datasets
    
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    
    c = {}
    counter = 0
    for dataset in list_datasets:
        c[dataset] = colors[counter]
        counter = counter + 1
        if counter >= len(colors):
            counter = 0

    plt.rcParams.update({'font.size': 15})
    ax = df_perc_all.plot(kind='bar', figsize=(15,10), color=c, width = 0.75)
    ax.set_ylabel("%")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.title("Value "+str(selected_value)+" in different datasets")
    
    for dataset in list_datasets:
        print("Number articles in "+str(dataset)+": "+str(len(df_with_topics[(df_with_topics['dataset'] == dataset) & (df_with_topics[value_int_list[0]] == 1)])))


def values_in_different_groups(df_with_topics, dict_anchor_words, selected_dataset):
         
    list_datasets = df_with_topics['dataset'].unique().tolist()

    df_with_topics_field = df_with_topics.loc[df_with_topics['dataset'] == selected_dataset]
        

    list_datasets = df_with_topics['dataset'].unique().tolist()

        
    df_sum_dataset_short = df_with_topics_field.sum(numeric_only=True)
               

        
    series_perc_dataset_short = df_sum_dataset_short.apply(lambda x: x / len(df_with_topics_field) * 100)
    series_perc_dataset_short = series_perc_dataset_short[:len(dict_anchor_words)]

    counter = 0
    for value, keywords in dict_anchor_words.items():
        series_perc_dataset_short = series_perc_dataset_short.rename({counter: value})
        counter = counter + 1
        
    series_perc_dataset_short = series_perc_dataset_short.sort_values(ascending = False)
        
    series_perc_dataset_short = series_perc_dataset_short.rename(selected_dataset)
    df_perc_dataset_short = series_perc_dataset_short.to_frame()

        
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    
    c = {}
    counter = 0
    for dataset in list_datasets:
        c[dataset] = colors[counter]
        counter = counter + 1
        if counter >= len(colors):
            counter = 0
        
    plt.rcParams.update({'font.size': 16})
    ax = df_perc_dataset_short.plot(kind='barh', figsize=(10,10),
                                        color=c[selected_dataset])
    ax.set_xlabel("%")
    plt.title("Distribution of values")
    plt.gca().invert_yaxis()

def topic_int_or_string(Topic_selected, dict_anchor_words):
    
    if type(Topic_selected) == str:
        list_keys = list(dict_anchor_words.keys())
        Topic_selected_number = list_keys.index(Topic_selected)
    else:
        Topic_selected_number = Topic_selected
        
    return Topic_selected_number

def create_vis_frequency_values(df_with_topics, dict_anchor_words):
    
    # list values and list values int
    name_values = list(dict_anchor_words.keys())
    list_values_int = []
    for i in name_values:
        integ = topic_int_or_string(i, dict_anchor_words)
        list_values_int.append(integ)

   
    df_with_topics_sum_dataset_short = df_with_topics[[c for c in df_with_topics.columns if c in list_values_int]]
    df_with_topics_sum_dataset_short.columns = name_values
    df_sum_dataset_short = df_with_topics_sum_dataset_short.sum(numeric_only=True)
    series_perc_dataset_short = df_sum_dataset_short.apply(lambda x: x / len(df_with_topics_sum_dataset_short) * 100)
    series_perc_dataset_short = series_perc_dataset_short.sort_values(ascending=False)
    
    df_perc_dataset_short = series_perc_dataset_short.to_frame()
    #df_perc_dataset_short.columns = ["Percentage of documents mentioning each value"]
    

    
#    c = {"NEWS": "#1f77b4", "ETHICS": "#ff7f0e", "TECH": "#2ca02c", "LEGAL": "#d62728"}
    
    plt.rcParams.update({'font.size': 16})
    ax = df_perc_dataset_short.plot(kind='bar', figsize=(6,6),legend=False)#,
#                                    color=c)
    ax.set_ylabel("%")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #plt.show()

def create_vis_values_over_time(df_with_topics, dict_anchor_words, resampling, values_to_include_in_visualisation, smoothing, max_value_y):
    
    copy_df_with_topics = df_with_topics.copy()
    copy_dict_anchor_words = dict_anchor_words.copy()
    
    df_with_topics_freq = copy_df_with_topics.set_index('date').resample(resampling).size().reset_index(name="count")
    df_with_topics_freq = df_with_topics_freq.set_index('date')

    df_frequencies = copy_df_with_topics.set_index('date')
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

def create_vis_value_over_time_in_diff_datasets(df_with_topics, selected_value, dict_anchor_words, resampling, values_to_include_in_visualisation, smoothing, max_value_y):
    
    values_to_include_in_visualisation.append(selected_value)
    copy_df_with_topics = df_with_topics.copy()
    copy_dict_anchor_words = dict_anchor_words.copy()
        
    list_datasets = df_with_topics.groupby(['dataset']).size().index.tolist()
    
    df_value_in_datasets = pd.DataFrame()
    df_counts = pd.DataFrame()
    for dataset in list_datasets:    
        
        copy_df_with_topics = df_with_topics.copy()
        copy_dict_anchor_words = dict_anchor_words.copy()
    
        copy_df_with_topics = copy_df_with_topics[copy_df_with_topics['dataset'] == dataset]
          
        df_with_topics_freq = copy_df_with_topics.set_index('date').resample(resampling).size().reset_index(name="count")
        df_with_topics_freq = df_with_topics_freq.set_index('date')
    
        df_frequencies = copy_df_with_topics.set_index('date')
        df_frequencies = df_frequencies.resample(resampling).sum()
            
        list_topics = list(range(len(copy_dict_anchor_words)))
        df_frequencies = df_frequencies[list_topics]
          
        df_frequencies = df_frequencies[list_topics].div(df_with_topics_freq["count"], axis=0)
        combined_df = pd.concat([df_frequencies, df_with_topics_freq], axis=1)
        combined_df = combined_df.fillna(0)
           
        name_values = list(copy_dict_anchor_words.keys())
        
        combined_df[list_topics] = combined_df[list_topics] * 100
        combined_df.columns = name_values + [str("count_"+str(dataset))]
        combined_df_value = pd.DataFrame(combined_df[selected_value])
        combined_df_value = combined_df_value.rename(columns={selected_value: dataset})
        df_value_in_datasets = pd.concat([df_value_in_datasets, combined_df_value], axis=1)
    
        df_counts_value = pd.DataFrame(combined_df[str("count_"+str(dataset))])
        df_counts = pd.concat([df_counts, df_counts_value], axis=1)
    
    x = pd.Series(df_value_in_datasets.index.values)
    x = x.dt.to_pydatetime().tolist()
    
    x = [ z - relativedelta(years=1) for z in x]
        
    sigma = (np.log(len(x)) - 1.25) * 1.2 * smoothing
    
    
    fig, ax1 = plt.subplots()
    for dataset in list_datasets:
            ysmoothed = gaussian_filter1d(df_value_in_datasets[dataset], sigma=sigma)
            ax1.plot(x, ysmoothed, label=str(dataset), linewidth=2)
        
    ax1.set_xlabel('Time', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Percentage of documents addressing each value \n per unit of time (lines)  (%)', fontsize=12, fontweight="bold")
    ax1.legend(prop={'size': 10})
        
    timestamp_0 = x[0]
    timestamp_1 = x[1]
        
    
    #width = (time.mktime(timestamp_1.timetuple()) - time.mktime(timestamp_0.timetuple())) / 86400 *.8
    width = (timestamp_1 - timestamp_0).total_seconds() / 86400 * 0.8
           
    ax2 = ax1.twinx()
    for dataset in list_datasets:
        ax2.bar(x, df_counts[str("count_"+str(dataset))].tolist(), width=width, alpha=0.1 #color='gainsboro'
              )
    ax2.set_ylabel('Number of documents in the dataset \n per unit of time (bars)', fontsize=12, fontweight="bold")
        
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
        
    ax1.set_ylim([0,max_value_y])
        
    
    fig.tight_layout() 
    plt.title("Change in the occurence of "+str(selected_value), fontsize=16, fontweight="bold")
    plt.figure(figsize=(20,14), dpi= 400)
    
    
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
    
    df_with_topics_freq_value_0 = copy_df_with_topics[[values_selected[0], 'date']].set_index('date').resample(resampling).size().reset_index(name="count")
    df_with_topics_freq_value_0 = df_with_topics_freq_value_0.set_index('date')
    
    df_with_topics_selected_topics = copy_df_with_topics[values_selected]
    list_counts = df_with_topics_selected_topics.sum(axis=1).tolist()
    
    counter = 0
    for i in list_counts:
        if i == len(values_selected):
            list_counts[counter] = 1
        else:
            list_counts[counter] = 0
        counter += 1
       
    df_with_topics_sum = copy_df_with_topics[["date"]]
    df_with_topics_sum = df_with_topics_sum.set_index('date')
    
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
    df_with_topics_selected_topic = df_with_topics_selected_topic.set_index('date')  
    
    df_with_topics_freq = df_with_topics_selected_topic.resample(resampling).size().reset_index(name="count")
    df_with_topics_freq = df_with_topics_freq.set_index('date')
    
    for word in list_words:
        df_with_topics_selected_topic[word] = df_with_topics_selected_topic["text"].str.contains(pat = word).astype(int) #''' Check here '''
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

def compare_words_topics_in_runs (dict_all_df_with_topics, dict_all_topics, dict_anchor_words, selected_value, model_runs_to_show, resampling, smoothing, max_value_y):
    
    copy_dict_anchor_words = dict_anchor_words.copy()

    combined_df = pd.DataFrame()
    names_topic_models = []
    dict_names_runs = {}

    for topic_model in model_runs_to_show:       
        copy_df_with_topics = dict_all_df_with_topics[topic_model]
        df_with_topics_freq = copy_df_with_topics.set_index('date').resample(resampling).size().reset_index(name="count")
        df_with_topics_freq = df_with_topics_freq.set_index('date')
    
        df_frequencies = copy_df_with_topics.set_index('date')
        
        df_frequencies = df_frequencies.resample(resampling).sum()
           
        list_values = list(copy_dict_anchor_words.keys())
        selected_value_int = list_values.index(selected_value)
        df_frequencies_2 = pd.DataFrame()
        df_frequencies_2['topic_model_'+str(topic_model)] = df_frequencies[selected_value_int].div(df_with_topics_freq["count"], axis=0)
        names_topic_models.append('topic_model_'+str(topic_model))
        dict_names_runs[topic_model]='topic_model_'+str(topic_model)
        combined_df = pd.concat([combined_df, df_frequencies_2], axis=1)
        if topic_model == 0:
            combined_df = pd.concat([combined_df, df_with_topics_freq], axis=1)
        
    combined_df.rename(columns = dict_names_runs, inplace = True)
    
    combined_df = combined_df.fillna(0)
    #print(combined_df)
        
    x = pd.Series(combined_df.index.values)
    x = x.dt.to_pydatetime().tolist()
    
    x = [ z - relativedelta(years=1) for z in x]
             
    combined_df[names_topic_models] = combined_df[names_topic_models] * 100
    
    sigma = (np.log(len(x)) - 1.25) * 1.2 * smoothing
     
    n_colors = len(names_topic_models)
    colours = cm.tab20(np.linspace(0, 1, n_colors)) 
    
    list_words_first_model = dict_all_topics[0][selected_value_int]
    counter = 0
    
    for topic_model in model_runs_to_show:
        dict_topics = dict_all_topics[counter]
        #dict_topics = eval(dict_topics)
        #exec('dict_topics = topics_'+str(counter))
        list_words = dict_topics[selected_value_int]
        words_in_color = list_words
        if counter > 0:
            for y in range(len(list_words)):
                if list_words[y] not in list_words_first_model:
                    list_words[y] = Fore.GREEN + str(list_words[y]) + Fore.RESET      
            words_in_color = "["+str(', '.join(str("'"+str(item)+"'") for item in list_words))+"]"
        print("\033[1m" + "Topic_model_" + str(topic_model)+"\033[0m" +": "+str(words_in_color))
        counter = counter + 1
    
    #print(dict_labels)
    #print(dict_labels[1])
        
    counter = 0
    fig, ax1 = plt.subplots()
    for topic_model in names_topic_models:
            ysmoothed = gaussian_filter1d(combined_df[topic_model].tolist(), sigma=sigma)
            ax1.plot(x, ysmoothed, label=str(topic_model), linewidth=2, color = colours[counter])
            counter = counter + 1
        
    ax1.set_xlabel('Time', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Percentage of documents addressing each value \n per unit of time (lines)  (%)', fontsize=12, fontweight="bold")
    ax1.legend()
        
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
    
    plt.rcParams["figure.figsize"] = [12,6]
    plt.show()
    
def intertopic_distance_map(df_with_topics, topics, list_topics_to_remove): 
    
    title: str = "<b>Intertopic Distance Map</b>"
    width: int = 650
    height: int = 650
   
    topic_list = list(df_with_topics.columns)
    topic_list = [num for num in topic_list if isinstance(num, (int,float))]
    df_only_topics = df_with_topics[topic_list]
    frequencies = df_only_topics.sum().tolist()
    
    for i in range(len(frequencies)):
        if frequencies[i] == 0 or frequencies[i] == len(df_only_topics):
            list_topics_to_remove.append(i)
    
    topic_list = [topic for topic in topic_list if topic not in list_topics_to_remove]
    df_only_topics = df_only_topics[topic_list]
    frequencies = df_only_topics.sum().tolist()
    
    embeddings = df_only_topics.T.to_numpy()
    embeddings = MinMaxScaler().fit_transform(embeddings)
    embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger', random_state=42).fit_transform(embeddings)
    
    words = []
    for i in topic_list:
        words.append(' | '.join(topics[i][:5]))
    
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                       "Topic": topic_list, "Words": words, "Size": frequencies})

    return _plotly_topic_visualization(df, topic_list, title, width, height)

def _plotly_topic_visualization(df: pd.DataFrame,
                                topic_list: List[str],
                                title: str,
                                width: int,
                                height: int):
    """ Create plotly-based visualization of topics with a slider for topic selection """

    def get_color(topic_selected):
        if topic_selected == -1:
            marker_color = ["#B0BEC5" for _ in topic_list]
        else:
            marker_color = ["red" if topic == topic_selected else "#B0BEC5" for topic in topic_list]
        return [{'marker.color': [marker_color]}]

    # Prepare figure range
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))

    # Plot topics
    fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
                     hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False})
    fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color='DarkSlateGrey')))

    # Update hover order
    fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
                                                 "%{customdata[1]}",
                                                 "Size: %{customdata[2]}"]))

    # Create a slider for topic selection
    steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    # Stylize layout
    fig.update_layout(
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        sliders=sliders
    )

    # Update axes ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    # Add grid in a 'plus' shape
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
    fig.data = fig.data[::-1]

    return fig

def perform_sentiment_analysis(df_with_topics, selected_value, dict_anchor_words, starttime, endtime):
    analyzer = SentimentIntensityAnalyzer()
    
    value_int = list(dict_anchor_words.keys()).index(selected_value)
    
    selected_dataset = df_with_topics.loc[(df_with_topics[value_int] == 1) & (df_with_topics['date'] >= dateutil.parser.parse(str(starttime))) & (df_with_topics['date'] < dateutil.parser.parse(str(endtime)))] 
    
    selected_dataset['polarity'] = selected_dataset['text'].apply(lambda x: analyzer.polarity_scores(x))
    selected_dataset = pd.concat([selected_dataset.drop(['polarity'], axis=1), selected_dataset['polarity'].apply(pd.Series)], axis=1)
    selected_dataset['sentiment'] = selected_dataset['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
    sns.countplot(y='sentiment', 
                 data=selected_dataset, 
                 palette=['#b2d8d8',"#008080", '#db3d13']
                 );
    plt.show()
    
    g = sns.lineplot(x='date', y='compound', data=selected_dataset)

    g.set(xticklabels=[]) 
    g.set(title='Sentiment of articles')
    g.set(xlabel="Time")
    g.set(ylabel="Sentiment")
    g.tick_params(bottom=False)

    g.axhline(0, ls='--', c = 'grey')
    plt.show()
    
    sns.boxplot(y='compound', 
            x='sentiment',
            palette=['#b2d8d8',"#008080", '#db3d13'], 
            data=selected_dataset);
    plt.show()


def inspect_words_over_time_based_on_most_frequent_words(df_with_topics, dict_anchor_words, model_and_vectorized_data, topic_to_evaluate, number_of_words, resampling, smoothing, max_value_y):
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
    list_words = list(list(zip(*model_and_vectorized_data[0].get_topics(topic=topic_to_evaluate_number, n_words=number_of_words)))[0])
    inspect_words_over_time(df_with_topics, topic_to_evaluate_number, list_words, resampling, smoothing, max_value_y)

def inspect_words_over_time_based_on_own_list(df_with_topics, dict_anchor_words, topic_to_evaluate, list_words, resampling, smoothing, max_value_y):
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
    inspect_words_over_time(df_with_topics, topic_to_evaluate_number, list_words, resampling, smoothing, max_value_y)




