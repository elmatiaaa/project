import sklearn
import streamlit as st
import pandas as pd 
import numpy as np 
import warnings
from sklearn.metrics import make_scorer, accuracy_score,precision_score
warnings.filterwarnings('ignore', category=UserWarning, append=True)

# data
df = pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/machine-Learning/main/mushrooms.csv")
df.head()

# normalisasi
# data yang dipakai 2000 data
# pemisahan class dan fitur
df=df[:2000]
from sklearn.preprocessing import OrdinalEncoder
x = df.drop(df[['class']],axis=1)
enc = OrdinalEncoder()
a = enc.fit_transform(x)
x=pd.DataFrame(a, columns=x.columns)

# class
y = df.loc[:, "class"]
y = df['class'].values

# Split Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

st.set_page_config(page_title="Ima")
@st.cache()
def progress():
    with st.spinner("Bentar ya....."):
        time.sleep(1)
        
st.title("UAS PENDAT")

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Jamur Data", "Prepocessing", "Modeling", "Implementation"])

with dataframe:
    st.write('Data Jamur')
    dataset,data= st.tabs(['Dataset',"data"])
    with dataset:
        st.dataframe(df)

        
with preporcessing:
    st.write('Ordinal Encoder')
    st.dataframe(x)

with modeling:
    # pisahkan fitur dan label
    knn,naivebayes,decisiontree= st.tabs(
        ["K-Nearest Neighbor","naivebayes","decisiontree"])
    with knn:
      from sklearn.neighbors import KNeighborsClassifier
      knn = KNeighborsClassifier(n_neighbors=3)
      knn.fit(x_train,y_train)
      y_pred_knn = knn.predict(x_test) 
      accuracy_knn=round(accuracy_score(y_test,y_pred_knn)* 100, 2)
      acc_knn = round(knn.score(x_train, y_train) * 100, 2)
      label_knn = pd.DataFrame(
      data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index()
      st.success(f'Tingkat akurasi = {acc_knn}')
      st.dataframe(label_knn)

    with naivebayes:
        #Metrics
        from sklearn.metrics import make_scorer, accuracy_score,precision_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

        #Model Select
        from sklearn.model_selection import KFold,train_test_split,cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import  LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import linear_model
        from sklearn.linear_model import SGDClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC, LinearSVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform👍
        gaussian = GaussianNB()
        gaussian.fit(x_train, y_train)
        y_pred = gaussian.predict(x_test) 
        accuracy_nb=round(accuracy_score(y_test,y_pred)* 100, 2)
        acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        precision =precision_score(y_test, y_pred,average='micro')
        recall =  recall_score(y_test, y_pred,average='micro')
        f1 = f1_score(y_test,y_pred,average='micro')
        print('Confusion matrix for Naive Bayes\n',cm)
        print('accuracy_Naive Bayes: %.3f' %accuracy)
        print('precision_Naive Bayes: %.3f' %precision)
        print('recall_Naive Bayes: %.3f' %recall)
        print('f1-score_Naive Bayes : %.3f' %f1)
        st.success(accuracy)
        label_nb = pd.DataFrame(
        data={'Label Test': y_test, 'Label Predict': y_pred})
        label_nb
        
        
    with decisiontree:
        from sklearn.tree import DecisionTreeClassifier
        d3 = DecisionTreeClassifier()
        d3.fit(x_train, y_train)
        y_predic = d3.predict(x_test)
        data_predic = pd.concat([pd.DataFrame(y_test).reset_index(drop=True), pd.DataFrame(y_predic, columns=["Predict"]).reset_index(drop=True)], axis=1)        
        from sklearn.metrics import accuracy_score
        a=f'acuraty = {"{:,.2f}".format(accuracy_score(y_test, y_predic)*100)}%'
        st.success(a)
        data_predic
        
with implementation:
        df=df[:2000]
        from sklearn.preprocessing import OrdinalEncoder
        x = df.drop(df[['class']],axis=1)
        enc = OrdinalEncoder()
        a = enc.fit_transform(x)
        x=pd.DataFrame(a, columns=x.columns)
        capshape=st.text_input('cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s')
        capsurface=st.text_input('cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s')
        capcolor=st.text_input('cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y')
        bruises=st.text_input('bruises: bruises=t,no=f')
        odor=st.text_input('odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s')
        gillattachment=st.text_input('gill-attachment: attached=a,descending=d,free=f,notched=n')
        gillspacing=st.text_input('gill-spacing: close=c,crowded=w,distant=d')
        gillsize=st.text_input('gill-size: broad=b,narrow=n')
        gillcolor=st.text_input('gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y')
        stalkshape=st.text_input('stalk-shape: enlarging=e,tapering=t')
        stalkroot=st.text_input('stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?')
        stalksurfaceabovering=st.text_input('stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s')
        stalksurfacebelowring=st.text_input('stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s')
        stalkcolorabovering=st.text_input('stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y')
        stalkcolorbelowring=st.text_input('stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y')
        veiltype=st.text_input('veil-type: partial=p,universal=u')
        veilcolor=st.text_input('veil-color: brown=n,orange=o,white=w,yellow=y')
        ringnumber=st.text_input('ring-number: none=n,one=o,two=t')
        ringtype=st.text_input('ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z')
        sporeprintcolor=st.text_input('spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y')
        population=st.text_input('population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y')
        habitat=st.text_input('habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d')
#x_new = ['x','y','y','t','l','f','c','b','g','e','c','s','s','w','w','p','w','o','p','k','s','m'] # hasil=0/e
        x_new = [capshape,capsurface,capcolor,bruises,odor,gillattachment,gillspacing,gillsize,gillcolor,stalkshape,stalkroot,stalksurfaceabovering,stalksurfacebelowring,stalkcolorabovering,stalkcolorbelowring,veiltype,veilcolor,ringnumber,ringtype,sporeprintcolor,population,habitat] # hasil=1/p
        hinput=enc.transform(np.array([x_new]))
        hinput
        clf_pf = GaussianNB()
        clf_pf.predict([hinput])
        #Metrics
        from sklearn.metrics import make_scorer, accuracy_score,precision_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

        #Model Select
        from sklearn.model_selection import KFold,train_test_split,cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import  LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import linear_model
        from sklearn.linear_model import SGDClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC, LinearSVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform👍
        gaussian = GaussianNB()
        gaussian.fit(x_train, y_train)
        y_pred = gaussian.predict(hinput) 
        st.write(y_pred)
