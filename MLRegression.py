
#Maskinlaering med lineaer regresjon
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm, cross_validation
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

pd.set_option('display.max_columns', None)

#Lager en dataframe fra quandl "WIKI/GOOGL" kansje litt rart att den ikke gaar fram til dagen idag
df = quandl.get("WIKI/GOOGL")

#Bruker kun angitte kolonner i en ny dataframe
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#Regner ut prosent av adjusted high - adjusted low kolonner / adjusted low * 100
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100

#Regner ut prosent av adjusted close - adjusted open kolonner / adjusted open * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#Lager en ny dataframe fra foelgene kolonner
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#Legger til en string i en variabel
forecast_col = 'Adj. Close'

#Legger inn -99999 paa alle steder der det er NaN, vanlig i flere maskinlaerings classer(de takler ikke NaN), blir ignorert
df.fillna(value=-99999, inplace=True)

#Legger inn tak paa forecast med en multiplier
forecast_out = int(math.ceil(0.01 * len(df)))

#Legger inn kolonne label i dataframe med forecast_out utregning
df['label'] = df[forecast_col].shift(-forecast_out)

#Dropper alle NaN som er igjen i dataframe
#df.dropna(inplace=True)

#Hopper ned til linje 97!

#Bruker alle kolonner utenom label som Features til maskinlaeringsmodellen
#X = np.array(df.drop(['label'], 1))

#Bruker liten y som label som Label i maskinlaeringsmodellen, bruke X og y her er normen
#y = np.array(df['label'])

#Preprocessering for aa lage verdiene om til fra -1 og til 1
#X = preprocessing.scale(X)

#Skjonner ikke hvorfor dette blir gjort 2 ganger
#y = np.array(df['label'])

#Splitter opp trainingdata og testdata?????
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#Classifier, her kan vi velge forskejllige maskinlaerings classifiere
'''A bit better, but basically the same. So how might we know, as scientists, 
which algorithm to choose? After a while, you will get used to what works in most situations 
and what doesn't. You can also check out: choosing the right estimator from scikit-learn's website. 
This can help you walk through some basic choices. If you ask people who use machine learning often though, 
it's really trial and error. You will try a handful of algorithms and simply go with the one that works best. 
Another thing to note is that some of the algorithms must run linearly, others not. Do not confuse linear regression with the 
requirement to run linearly, by the way. So what does that all mean? Some of the machine learning algorithms here will process 
one step at a time, with no threading, others can thread and use all the CPU cores you have available. 
You could learn a lot about each algorithm to figure out which ones can thread, or you can visit the documentation, 
and look for the n_jobs parameter. If it has n_jobs, you have an algorithm that can be threaded for high performance. 
If not, tough luck! Thus, if you are processing massive amounts of data, or you need to process medium data but at a very 
high rate of speed, then you would want something threaded. Let's check for our two algorithms.

Heading to the docs for sklearn.svm.SVR, and looking through the parameters, do you see n_jobs? Not me. So no, no threading here. 
As you could see, on our small data, it makes very little difference, but, on say even as little as 20mb of data, it makes a massive 
difference. Next up, let's check out the LinearRegression algorithm. Do you see n_jobs here? Indeed! So here, you can specify exactly 
how many threads you'll want. If you put in -1 for the value, then the algorithm will use all available threads.'''
#clf = LinearRegression(n_jobs=-1)

#Legger inn X(features) og y(labels) training data inn i classifier og trener den opp
#clf.fit(X_train, y_train)

#Tester accuracy imot X(features) og y(labels) test variablene
#accuracy = clf.score(X_test, y_test)

#Vi velger LinearRegression fra sklearn fordi den gav oss hoyest accuracy
'''
#Tester forskjellig typer kernel i svm.SVR classifier
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(k, accuracy)
'''

#Husk aa les og studer resten av dette scriptet paa pythonprogramming.net

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

#Stilen som matplotlib bruker
style.use('ggplot')

#Legger inn en ekstra kolonne i dataframe som heter Forecast og legger inn Nan(Not a Number) som verdier
df['Forecast'] = np.nan

#Henter inn datoer for plotting i matplotlib
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#Legger inn ekstra datoer for de som er "Forecast" i dataframe for plotting
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#Plotter data til graf med matplotlib
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()