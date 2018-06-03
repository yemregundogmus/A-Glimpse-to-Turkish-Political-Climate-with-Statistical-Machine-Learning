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


print("Siyasi Analizciye Hoşgeldiniz")
print("\nBurada Size Birkaç Soru Sorup Bunların Karşılığında sizin Siyasi Görüşünüzü Tahmin Edeceğiz\n")

soru1 = input("Cinsiyetiniz Erkek için 0 Kadın için 1 Yazınız. :")
soru2 = input("\nYaşınız 0-18 için 1 18-30 için 2 30-50 için 3 50-60 için 4 60+ 5 Yazınız. :\n")
soru3 = input("\nYaşadığınız Bölge Marmara İçin 1 Ege İçin 2 Karadeniz İçin 3 Akdeniz İçin 4 İç Anadolu İçin 5 Doğu Anadolu İçin 6 Güneydoğu İçin 7 Yazınız. :\n")
soru4 = input("\nEğitim Seviyeniz İlkokul İçin 1 Ortaokul İçin 2 Lise İçin 3 Ön Lisans İçin 4 Lisans İçin 5 Lisans Üstü iİçin 6 Yazınız. :\n")
soru5 = input("\nEkonomik Durumumuz'un iyi olduğunu düşünüyor musunuz? Evet İçin 1 Hayır için 0 :\n")
soru6 = input("\nEğitimde Reform Gerekiyor mu? Evet İçin 1 Hayır için 0 :\n")
soru7 = input("\nÖzelleştirmeye Karşı Çıkar Mısınız? Evet İçin 1 Hayır için 0 :\n")
soru8 = input("\nDevlet idam gibi bir cezayı belirli suçlara karşı kullanmalı mıdır? Evet İçin 1 Hayır için 0 :\n")
soru9 = input("\nGazetecilerimizi Yeterince Tarafısız Buluyor Musunuz? Evet İçin 1 Hayır için 0 :\n")
soru10 = input("\n22:00 'dan Sonra İçki Yasağını Destekliyor Musunuz? Evet İçin 1 Hayır için 0 :\n")
soru11 = input("\nDaha Laik Bir Devlette Yaşamak İster Misiniz? Evet İçin 1 Hayır için 0 :\n")
soru12 = input("\nKürtaj Yasağını Destekliyor Musunuz? Evet İçin 1 Hayır için 0 :\n")
soru13 = input("\nOhal'in Özgürlükleri Kısıtladığını Düşünüyor Musunuz? Evet İçin 1 Hayır için 0 :\n")
soru14 = input("\nMeclise Yeni Bir Partinin Girmesini İster Misiniz? Evet İçin 1 Hayır için 0 :\n")

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
    print("AKP'ye Yakınsınız")
if (y_pred == 1):
    print("İYİ'ye Yakınsınız")
if (y_pred == 2):
    print("TARAFSIZ'sınız")
if (y_pred == 3):
    print("HDP'ye Yakınsınız")
if (y_pred == 4):
    print("MHP'ye Yakınsınız")
if (y_pred == 5):
    print("CHP'ye Yakınsınız")

