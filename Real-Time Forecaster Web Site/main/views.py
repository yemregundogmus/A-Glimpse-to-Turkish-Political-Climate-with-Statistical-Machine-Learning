from django.shortcuts import render, HttpResponseRedirect

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create your views here.


def home_view(request):

    return render(request, "index.html", {})

def sonuc_view(request):

    if request.method != "POST":
        return HttpResponseRedirect("/")


    soru1 = request.POST.get("soru1")
    soru2 = request.POST.get("soru2")
    soru3 = request.POST.get("soru3")
    soru4 = request.POST.get("soru4")
    soru5 = request.POST.get("soru5")
    soru6 = request.POST.get("soru6")
    soru7 = request.POST.get("soru7")
    soru8 = request.POST.get("soru8")
    soru9 = request.POST.get("soru9")
    soru10 = request.POST.get("soru10")
    soru11 = request.POST.get("soru11")
    soru12 = request.POST.get("soru12")
    soru13 = request.POST.get("soru13")
    soru14 = request.POST.get("soru14")

    try:
            # Import Data
        data = pd.read_csv('yonelimfinal.csv')

        # Delete irrelevant feature
        data = data.drop(['Timestamp'], axis = 1)
        # Rename Columns
        names = ["Sex","Age","Region","Education","Economy_Good","Education_Reform",
                 "Against_Denationalization","Support_Death_Penalty","Obejctive_Journalists","Alcohol_Prohibition",
                 "More_Secular","Abortion_Ban","Restricted_Freedoms","Support_New_Party"]
        data = data.rename(columns=dict(zip(data.columns, names)))
        # Give Discrete values to questionn answers yes/no
        question_mapping = {"Evet": 1, "Hayır": 0}
        for i in range(4,14):
            data[names[i]] = data[names[i]].map(question_mapping)

        # Mapping Discrete Values to categorical data
        opinion_mapping = {"AKP": 0, "IYI": 1, "IYI PARTI": 1, "DIĞER" : 2, "OTHS" : 2, "HDP" : 3, "MHP" : 4 ,"CHP": 5}   
        city_mapping = {"Marmara": 1, "Ege": 2, "Karadeniz": 3, "Akdeniz": 4, "İç Anadolu": 5, "Doğu Anadolu": 6, "Güneydoğu": 7}
        age_mapping = {"0-18": 1, "18-30": 2, "30-50": 3, "50-60": 4, "60+": 5}
        sex_mapping = {"Erkek": 0, "Kadın": 1}
        education_mapping = {"İlkokul": 1, "Ortaokul": 2, "Lise": 3, "Ön Lisans": 4, "Lisans": 5, "Lisans Üstü": 6}

        # Apply mappinng
        def mymap(x, mapping): return mapping[x]
        data['parti'] = data['parti'].apply(mymap, mapping = opinion_mapping)
        data['Region'] = data['Region'].apply(mymap, mapping = city_mapping)
        data['Age'] = data['Age'].apply(mymap, mapping = age_mapping)
        data['Sex'] = data['Sex'].apply(mymap, mapping = sex_mapping)
        data['Education'] = data['Education'].apply(mymap, mapping = education_mapping)

        def getOpinion(data, party = None):
            if party != None: 
                data = data[(data.parti == opinion_mapping[party])]
                print("Opinion of ", party)
            return [data[col].mean() for col in names[4:]]



        dataCHP = data[(data.parti == opinion_mapping['CHP'])]
        dataAKP = data[(data.parti == opinion_mapping['AKP'])]
        dataIYI = data[(data.parti == opinion_mapping['IYI PARTI'])]
        dataMHP = data[(data.parti == opinion_mapping['MHP'])]
        dataDIGER = data[(data.parti == opinion_mapping['OTHS'])]
        dataHDP = data[(data.parti == opinion_mapping['HDP'])]

        data = pd.concat([dataCHP, dataIYI, dataAKP, dataMHP, dataDIGER, dataHDP])
        #data = pd.concat([dataIYI.sample(len(dataCHP), random_state=39), dataCHP])
        #data = pd.concat([dataIYI.sample(len(dataAKP), random_state=39), dataAKP])
        #data = pd.concat([dataCHP.sample(len(dataAKP), random_state=39), dataAKP, dataIYI.sample(len(dataAKP))])
        #data = pd.concat([dataCHP.sample(len(dataAKP), random_state=39), dataAKP, (dataIYI.sample(len(dataAKP)),dataIYI, (dataMHP.sample(len(dataAKP)),dataMHP, dataDIGER.sample(len(dataAKP), dataDIGER,dataHDP.sample(len(dataAKP)),dataHDP])

        predictors = data.drop(['parti'], axis=1)
        target = data["parti"]
        X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.44, random_state= 0)

        gbk = GradientBoostingClassifier()
        gbk.fit(X_train, y_train)
        y_pred = gbk.predict(X_train)
        acc_gbk= round(accuracy_score(y_pred, y_train) * 100, 2)

        xtest_new = np.array([soru1,soru2,soru3,soru4,soru5,soru6,soru7,soru8,soru9,soru10,soru11,soru12,soru13,soru14])
        xtest_new = xtest_new.reshape(1, -1)

        y_pred = gbk.predict(xtest_new)

        if (y_pred == 0):
            sonuc = "akp"
        if (y_pred == 1):
            sonuc = "iyi"
        if (y_pred == 2):
            sonuc = "tarafsiz"
        if (y_pred == 3):
            sonuc = "hdp"
        if (y_pred == 4):
            sonuc = "mhp"
        if (y_pred == 5):
            sonuc = "chp"

    except:
        sonuc = "hata"
        print("hataaa")



    return render(request, "sonuc.html", {'sonuc': sonuc})