from __future__ import unicode_literals# json for converting string dictionary from logout to dictionary
from django.shortcuts import render
import io
import matplotlib.pyplot as plt
import numpy as np
import urllib,base64
from urllib.parse import urlparse
#ALL THE MODEL LIBRARIES
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle  #for using previous tokenizer obj (The pickle module implements binary protocols for serializing and de-serializing a Python object structure.)
from tensorflow.keras.models import load_model

# gets plot input and further send website as a encoded string
def Give_url(figure):
    buf = io.BytesIO() #creating i/o stream(bcoz image is convert into binary and then sent to web as string)
    figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read()) #for encoding
    uri = urllib.parse.quote(string)
    return uri


missing_values = ["DATA EXPIRED", "CONSENT REVOKED"]
df = pd.read_csv("static/models/hell.csv", na_values=missing_values)  # OUR DATASET
df.drop(df.columns[0], axis=1, inplace=True)  #for column(removed serial number)
df.dropna(axis=0, inplace=True)  #for row

# IMPORT THE MODEL

#for giving one word
single = load_model("static/models/first_model_feeling1longonly_NEWMAIN.hdf5")

#for giving percentage to each feeling
MULTI_model = load_model("static/models/multi_traget_feeling.hdf5")

# TOKENIZOR OBJECT IMPORTED FROM FILE TO CONVERT STRING INPUT TO INTEGER TOKENS AS USED
# WHILE TRAINING GIVE PATH OF (token_NEWMAIN.pickle) FILE BELOW
with open('static/models/token_NEWMAIN.pickle', 'rb') as handle:
    tokenizortest = pickle.load(handle)


input = [] #for storing tokenized sentences

for i in df["text_long"]: #for reading each column of text //"text_long colum name has tweets"
    input.append(tokenizortest.texts_to_sequences([i])[0])

TOKENIZED_INPUT = np.array(pad_sequences(input, maxlen=90, padding='pre'))  #as neural network should have same length input
# created np arrayin above code


# BELOW CODE TO MAKE DICTIONARY TO INVERSE TRANSFORM PREDICTED RESULT
emotion_output = single.predict(TOKENIZED_INPUT) # used model for prediction
unique_emotion, ids = np.unique(np.array(df["chosen_emotion"]), return_inverse=True) #encoded to integer for fitting in model  # chosen emotuon tweet with their emotion # id stores code  #return inverse= true for decoding from string to int to string
unique_emotion = unique_emotion[emotion_output.argmax(1)] #for encoding so that int converted to string

unique, counts = np.unique(unique_emotion, return_counts=True)  # unique emotions haas strings
# DIFFERENT EMOTIONS OF WHOLE ANALYSIS
plt.figure() #for setting size
plt.bar(unique, counts)
plt.grid()
fig = plt.gcf() #for making 2d plots
Diff_emotions=Give_url(fig)   #fn gives encoded string and shows as plot on page

#Pie chart plot
plt.figure()
plt.axis('equal')  # for equal axis length
plt.pie(counts, labels=unique, autopct='%1.2f%%') #autopct for precision
fig = plt.gcf()
Piechart = Give_url(fig)

# HAPPINESS INDEX OF ALL COUNTRIES
data = df[["Nationality", "happiness"]] #each country haviong diff  percentage of happiness

data = data.groupby(['Nationality']).mean().reset_index() #n reset index is for storing mean valus in happiness column
data["happiness"] = data["happiness"].round(1) # rounding off
nationality = np.array(data["Nationality"])  # data is dataframe here converted to np
haapy_index = np.array(data["happiness"])
plt.figure(figsize=(15, 20)) # fig size of bar graph size
plt.barh(nationality, haapy_index, alpha=0.5, color='green') #alpha is opacity
plt.title("Happiness Index Of Countries", fontsize=30)
plt.yticks(nationality, fontsize=18)
plt.xticks(np.arange(0, 10, 1), fontsize=20) #ticks are values 1, 2,...
plt.xlabel("Happiness Index", fontsize=27)
plt.grid()
for index, value in enumerate(haapy_index):    #enumerate means traversing
    plt.text(value, index, str(value), fontsize=17)
fig = plt.gcf()
Happiness_index = Give_url(fig) # for transferring bar graph to website

'''
input = []
for i in df["text_long"]:
    input.append(tokenizortest.texts_to_sequences([i])[0])
TOKENIZED_INPUT = np.array(pad_sequences(input, maxlen=561, padding='pre'))
# BELOW CODE TO MAKE DICTIONARY TO INVERSE TRANSFORM PREDICTED RESULT
emotion_output = single.predict(TOKENIZED_INPUT)
unique_emotion, ids = np.unique(np.array(df["chosen_emotion"]), return_inverse=True)
unique_emotion = unique_emotion[emotion_output.argmax(1)]

unique, counts = np.unique(unique_emotion, return_counts=True)

# DIFFERENT EMOTIONS OF WHOLE ANALYSIS
plt.figure()
plt.bar(unique, counts)
plt.grid()
fig = plt.gcf()
Diff_emotions=Give_url(fig)

#Pie chart plot
plt.figure()
plt.axis('equal')
plt.pie(counts, labels=unique, autopct='%1.2f%%')
fig = plt.gcf()
Piechart = Give_url(fig)

# HAPPINESS INDEX OF ALL COUNTRIES
data = df[["Nationality", "happiness"]]

data = data.groupby(['Nationality']).mean().reset_index()
data["happiness"] = data["happiness"].round(1)
nationality = np.array(data["Nationality"])
haapy_index = np.array(data["happiness"])
plt.figure(figsize=(15, 20))
plt.barh(nationality, haapy_index, alpha=0.5, color='green')
plt.title("Happiness Index Of Countries", fontsize=30)
plt.yticks(nationality, fontsize=18)
plt.xticks(np.arange(0, 10, 1), fontsize=20)
plt.xlabel("Happiness Index", fontsize=27)
plt.grid()
for index, value in enumerate(haapy_index):
    plt.text(value, index, str(value), fontsize=17)
fig = plt.gcf()
Happiness_index = Give_url(fig)
'''

# bcoz model is trained by binary entropy so it doesnt accept decimal values
# CONVERT EACH EMOTION COLUMN TO BINARY CATEGORICAL VALUE
uniques_worry, ids_worry = np.unique(np.array(df["worry"]), return_inverse=True)

uniques_anger, ids_anger = np.unique(np.array(df["anger"]), return_inverse=True)

uniques_fear, ids_fear = np.unique(np.array(df["fear"]), return_inverse=True)

uniques_disgust, ids_disgust = np.unique(np.array(df["disgust"]), return_inverse=True)

uniques_anxiety, ids_anxiety = np.unique(np.array(df["anxiety"]), return_inverse=True)

uniques_sadness, ids_sadness = np.unique(np.array(df["sadness"]), return_inverse=True)

uniques_happiness, ids_happiness = np.unique(np.array(df["happiness"]), return_inverse=True)

uniques_relaxation, ids_relaxation = np.unique(np.array(df["relaxation"]), return_inverse=True)

uniques_desire, ids_desire = np.unique(np.array(df["desire"]), return_inverse=True)



# for loading page and above one are global variable so page will load fast as already those variables were made
def Index(request):


    if(request.method=='POST'):    # when user gives input
        MAIN_INPUT=request.POST.get('Input_tweet2')  # 'Input_tweet2' is id for that text area
       # MAIN_INPUT=MAIN_INPUT+"happy"
        print("MAIN_INPUT:"+MAIN_INPUT) # will be printed on terminal
        print("INPUT FROM THE PAGE TWEET HAS BEEN")
    else:
        MAIN_INPUT = "tweet whoes emotion we have to find happy so much" # this is for graph in which user gives input

    input = []
    input.append(tokenizortest.texts_to_sequences([MAIN_INPUT])[0])
    TOKENIZED_INPUT = np.array(pad_sequences(input, maxlen=90, padding='pre'))

    # BELOW CODE TO MAKE DICTIONARY TO INVERSE TRANSFORM PREDICTED RESULT
    uniques, ids = np.unique(df["chosen_emotion"], return_inverse=True)

    # PERFORM PREDICTION AND TRANSFORM RESULT to EMOTION STRING
    PREDICT_EMOTION = single.predict(TOKENIZED_INPUT)
    # BELOW LINE TRANSFORM INTEGER OUTPUT TO EMOTION
    OUT = uniques[PREDICT_EMOTION.argmax(1)]
    OUT=OUT[0] #for accessing that single value
    print("INPUT:" + str(tokenizortest.sequences_to_texts(TOKENIZED_INPUT)) + " OUTPUT" + str(OUT))
# for printing in terminal

    ##########################################################################
    # THE OUTPUT WILL BE IN BELOW ORDER
    # [worry,anger,disgust,fear,anxiety,sadness,happiness,relaxation,desire]
    ##########################################################################
    TOKENIZED_INPUT1 = np.array(pad_sequences(input, maxlen=60, padding='pre'))
    worry, anger, disgust, fear, anxiety, sadness, happiness, relaxation, desire = MULTI_model.predict(TOKENIZED_INPUT1)
    worry = uniques_worry[worry.argmax(1)] #for reverse transform
    anger = uniques_anger[anger.argmax(1)]
    disgust = uniques_disgust[disgust.argmax(1)]
    fear = uniques_fear[fear.argmax(1)]
    anxiety = uniques_anxiety[anxiety.argmax(1)]
    sadness = uniques_sadness[sadness.argmax(1)]
    happiness = uniques_happiness[happiness.argmax(1)]
    relaxation = uniques_relaxation[relaxation.argmax(1)]
    desire = uniques_desire[desire.argmax(1)]

    plt.figure(figsize=(15, 15))
    # model gives individual array
    Feeling = ["worry", "anger", "disgust", "fear", "anxiety", "sadness", "happiness", "relaxation", "desire"]
    Feeling_index = [worry[0], anger[0], disgust[0], fear[0], anxiety[0], sadness[0], happiness[0], relaxation[0],
                     desire[0]]
    plt.bar(Feeling, Feeling_index, alpha=0.5, color='grey')
    plt.title("Feeling Index Of Tweet", fontsize=23)
    plt.xticks(Feeling, fontsize=18)
    plt.yticks(np.arange(0, 11, 1), fontsize=15)
    plt.xlabel("Feelings ", fontsize=20)
    plt.ylabel("Scale", fontsize=20)
    plt.grid()
    for index, value in enumerate(Feeling_index):
        plt.text(index, value, str(value), fontsize=17)
    fig = plt.gcf()
    Tweet_Sentiment = Give_url(fig)

# render/run/ create html page with image url
    return render(request, "TrialAnaly.html", {"Diff_emotions": Diff_emotions,"Piechart":Piechart,"Happiness_index":Happiness_index,
                                             "Tweet_Sentiment":Tweet_Sentiment,"Output_Tweet":OUT})

#diff emotions is url or encoded string

#made by own