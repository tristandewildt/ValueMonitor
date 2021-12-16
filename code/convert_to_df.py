# Here it should first check what type of file it is. 
#If it is not one of the format, it should return that the format is wrong. 
import os
import pandas as pd
import re
import nltk
nltk.download('punkt')
import dateutil.parser
import rispy
from RISparser import readris


from os import path
#from nltk.tokenize import word_tokenize

def convert_from_csv_or_xlsx_to_df(datafile, extension):
    if extension == '.csv':
        df = pd.read_csv(datafile, encoding = "ISO-8859-1")
    if extension == '.xlsx':
        df = pd.read_excel(datafile) 

    
    ''' Check if strange characters before and after brackets'''
    for column in df:
        if "[" in column:
            new_name = column[column.index("["):]
            new_name = new_name[:column.index("]")]
            if "]" not in new_name:
                new_name = new_name + "]"
            df=df.rename(columns = {column:new_name})
    
    
    text_for_analysis_cols = [col for col in df.columns if '[Text_for_analysis]' in col]
    Other_information_cols = [col for col in df.columns if (col.startswith('[') and col.endswith(']') and '[Text_for_analysis]' not in col and '[Date]' not in col)] 
    
    df['combined'] = df[text_for_analysis_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df2 = df[['combined', '[Date]'] + Other_information_cols]
    df2.columns = ['[Text_for_analysis]', '[Date]'] + Other_information_cols
    return df2  
    
def convert_from_txt_to_df(datafile, extension):
    
    with open(datafile, 'r', errors='replace') as f:
        paras = f.readlines()
        paras = [x for x in paras if x != '\n']
        
    list_texts = []
    list_dates = []
    current_text = []
    current_date = []
    for para in paras:
        if para.startswith('[New_document]') and len(current_text) != 0:
            current_text = " ".join(current_text)
            list_texts.append(current_text)
            list_dates.append(current_date[0])
            current_text = []
            current_date = []
        
        if para.startswith('[Date]'):
            para = para.replace("[Date]","")
            para = para.replace("\n","")
            current_date.append(para)
                       
        else:
            para = para.replace("[New_document]","")
            para = para.replace("\n","")
            current_text.append(para)
    
    current_text = " ".join(current_text)
    list_texts.append(current_text)
    list_dates.append(current_date[0])
    
    df = pd.DataFrame(list(zip(list_texts, list_dates)), 
               columns =['[Text_for_analysis]', '[Date]'])    
    return df

def convert_to_df(file_name):
    #root = '/gdrive/My Drive/Topic_modelling_analysis/'
    root = os.getcwd()

    datafile = root + '/data/' + file_name  
    print(datafile)

    name, extension = os.path.splitext(file_name)
    
    accepted_formats = ['.csv', '.xlsx', '.txt', '']
        
    if path.exists(datafile) == False:
        raise ValueError("File has not been found. Check file name and if the file has been placed in the 'data' folder.")
        
    if extension not in accepted_formats:
        raise ValueError("Input file has the wrong format. Please use csv, xlsx or txt file.")
        
    if extension == '.csv' or extension == '.xlsx':
        df = convert_from_csv_or_xlsx_to_df(datafile, extension)  
        
    if extension == '.txt':
        df = convert_from_txt_to_df(datafile, extension)
        
    if extension == '':
        df = pd.read_pickle(datafile)
        df["[Text_for_analysis]"] = df["Title"] + ' ' +  df["Body"]
        df = df[['Date', '[Text_for_analysis]', 'Source', 'Country']]
        df = df.rename(columns={'Date': '[Date]'})
        df['[Date]'] = df['[Date]'].map(lambda x: dateutil.parser.parse(str(x)))
            
    ''' Here we clean the text and tokenize it'''
    df['[Text_for_analysis]'] = df['[Text_for_analysis]'].map(lambda x: re.sub(r'\W+', ' ', x))
    df['[Text_for_analysis]'] = df['[Text_for_analysis]'].map(lambda x: re.sub(r'http\S+', ' ', x))
    df['[Text_for_analysis]'] = df['[Text_for_analysis]'].map(lambda x: nltk.word_tokenize(x))
    df['[Text_for_analysis]'] = df['[Text_for_analysis]'].map(lambda x: " ".join(x))
        
    '''  Here we translate the dates to something workable '''  
    df['[Date]'] = df['[Date]'].map(lambda x: dateutil.parser.parse(str(x)))
    
    return df




def import_file_and_show_columns(corpus, file_format):
    file_format = file_format.lower()
    
    try:
        if file_format == 'csv':
            #df = pd.read_csv(corpus, encoding = "ISO-8859-1")
            df = pd.read_csv(corpus, encoding = 'utf-8-sig')
        if file_format == 'xlsx':
            df = pd.read_excel(corpus) 
        if file_format == 'json':
            df = pd.read_json(corpus)
    #    if file_format == 'ris':
    #        with open(corpus, 'r', encoding="utf8", errors='ignore') as bibliography_file:
    #            df = pd.DataFrame(rispy.load(bibliography_file))
        if file_format == 'pandas_df':
            df = pd.read_pickle(corpus)
    except ValueError:
        print("Error: check that the format of the file you provided is an accepted one, or that the variable 'file_format' matches the type of file provided as input")
      
    '''
    Possible errors:
    - if not written well 
    - if inconherence file_format
    - if file is not found
    - if file type is not one of the accepted ones
    '''
    
    print(df.info())
    return df


def prepare_df(df, list_columns):
    
    print(list_columns)
    
    df = df.fillna('')
    # combine columns
    text_cols = list_columns[0]
    #df['Text'] = df[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df['Text'] = df[text_cols].agg(' '.join, axis=1)
    df=df.rename(columns = {text_cols[1][0]:'Date'})
    
    df2 = df[['Text', 'Date']]
 #   df2.columns = ['[Text_for_analysis]', '[Date]']
    return df2 
    # add date and remove all other
    
    
    
    # process texts (make sure all text is set to low from here
    
    
    
    
   


def show_columns(corpus):
    
    '''first need to check what type of file this is'''

    if type(corpus) == dict:
        df = pd.DataFrame.from_dict(corpus)
    else:
        df = corpus
   
    print(df.info())

#file_name = "scopus_nucl_energy.csv"
#file_name = "scopus_nucl_energy.xlsx"
#file_name = "Covid_data.txt"
#root = '/gdrive/My Drive/Topic_modelling_analysis/'


#convert_to_df(file_name)



