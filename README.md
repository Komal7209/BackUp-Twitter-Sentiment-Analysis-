# Covid19TwitterSentiment-Analysis
In this project we had worked for Covid19 Twitter Sentiment Analysis. <br />
The dataset which we used to train our data consisted of tweet along with the original user sentiment at time of tweet. From a survey it also consisted of time taken for writing tweet and  1-10 scale for each individual feeling used


##  Prerequisite:

### Present Work
1. Knowledge of Django (This project used Django as framework)
2. Knowledge of RNN (Bidirectional LSTM) Algorithm
3. Knowledge of NLP
 
### Further modifications:
4. Twitter developer id (for live sentiment analysis)
5. Knowledge of IBM Cloud (for deployment)

## Dataset link : https://arxiv.org/pdf/2004.04225.pdf

 <br />
 
# Code part


 <br />
 
## 1. For installing libraries:   
<br />
conda install -c anaconda pillow  <br />
conda install -c conda-forge matplotlib  <br />
conda install -c anaconda seaborn  <br />
conda install tensorflow  <br />


##  For running server: 
<br />
python manage.py runserver  <br />

###  Frontend Backend integration
~ using function </br>
~ connections made and call in url.py  </br>
~ functions in views.py  </br>
~ url activate in html file  </br>

html(activation of function)-> url(cheking function) -> views.py(checking function definition)


# Content

### Frontend content :
                      Main->Static Folder-> Trial Analysis2
                      Main-> Templates->Html 
### Django files:      
                   Main-> Twwet_Dashboard-> models.py->(forms)
                                           urls.py->(paths for webpages)
                                           views.py-> (working functions of backend)
                                           Management-> commands-> bluemix_init.py(for hosting) 
                                           
### Jupyter notebooks: 
                       Jupyter_notebooks->Multifeeling_value.ipynb( For multiple sentences/lines output)
                                        ->twwet_me.ipynb ( For single line output )
                                     
### for preprocessing 
                       tokenizer-> tokenizer_SAVED_OOV.pickle   (tokenize (oov token)) 
                       (padding fn)[in jupyter notebook]
                       
### For dataset:
                 dataset->hell.csv
### For weights:
                  weights-> multi_traget_feeling.hdf5 (multiple feelings for a particular text as outcome as bar graph in dashboard)
                         -> first_model_feeling1longonly_NEWMAIN.hdf5(single feeling as a summary of multiline text)
## Some Documents:

Documentation link : https://drive.google.com/file/d/1E4MIv14svusdBCJkg6E3bNFCsnhkysSj/view?usp=sharing

PPT Link: https://drive.google.com/file/d/1fyJgoPZ6R57VXBwPeVX8mYqay2lOtYBV/view?usp=sharing

Video Link : https://drive.google.com/file/d/1LYOSQZQHyf8ZVZgsoek9iCeLQJCBde99/view?usp=sharing 

Sample text to be test: https://drive.google.com/file/d/1D_1HkI-xMGVw1PotbrnCsSswF8Byvbis/view?usp=sharing

All Files link : https://drive.google.com/drive/folders/1PzMCkXa3VQy1cj36E2ulXMNXDTC2R6jk?usp=sharing


## Research Papers for documentation
1. http://www.cs.columbia.edu/~julia/papers/Agarwaletal11.pdf    </br>
2. https://towardsdatascience.com/twitter-sentiment-analysis-based-on-news-topics-during-covid-19-c3d738005b55   </br>
3. https://arxiv.org/pdf/2003.05004    </br>
4. https://www.jmir.org/2020/4/e19016  </br>
5. https://arxiv.org/pdf/2003.10359 </br>
6. https://www.researchgate.net/profile/Kia_Jahanbin2/publication/339770709_Using_twitter_and_web_news_mining_to_predict_COVID-19_outbreak/links/5e84d4db4585150839b508b7/Using-twitter-and-web-news-mining-to-predict-COVID-19-outbreak.pdf </br>
7. https://arxiv.org/pdf/2003.12309 </br>
8. https://arxiv.org/pdf/2004.04225 </br>
9. https://towardsdatascience.com/how-are-americans-reacting-to-covid-19-700eb4d5b597?source=rss----7f60cf5620c9---4 </br>
