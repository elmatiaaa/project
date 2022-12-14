import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings("ignore")


st.title("PENAMBANGAN DATA C")

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("##### Nama  : Elmatia Dwi Uturiyah")
    st.write("##### Nim   : 200411100113 ")
    st.write("##### Kelas : Penambangan Data C ")
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : (Mobile Price Classification) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/code/akshitmadan/mobile-price-classification-knn/data")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1. id: Pengenalan
    """)
    st.write("""2. battery power : Baterai power Total energi yang dapat disimpan baterai dalam satu waktu diukur dalam mAh
    """)
    st.write("""3. bluethooth : bluethooth adalah spesifikasi industri untuk jaringan kawasan pribadi (personal area networks atau PAN) tanpa kabel. Bluetooth menghubungkan dan dapat dipakai untuk melakukan tukar-menukar informasi di antara peralatan-peralatan.
    """)
    st.write("""4. clock speed : kecepatan di mana mikroprosesor mengeksekusi instruksi
    """)
    st.write("""5. dual sim : Apakah mendukung dual sim atau tidak
    """)
    st.write("""6. fc : Kamera depan mega piksel
    """)
    st.write("""7. four_g : Apakah 4G atau tidak
    """)
    st.write("""8. int_memory : Memori Internal dalam Gigabyte
    """)
    st.write("""9. m_dep : Kedalaman Seluler dalam cm
    """)
    st.write("""10. mobile_wt : Berat ponsel
    """)
    st.write("""11. n_cores : Jumlah inti prosesor
    """)
    st.write("""12. pc: Mega piksel Kamera Utama
    """)
    st.write("""13. px_height : Tinggi Resolusi Piksel
    """)
    st.write("""14. px_width : Lebar Resolusi Piksel
    """)
    st.write("""15. ram : Memori Akses Acak dalam Megabita
    """)
    st.write("""16. sc_h : Tinggi layar ponsel dalam cm
    """)
    st.write("""17. sc_w : Lebar layar ponsel dalam cm
    """)
    st.write("""18. talk time : waktu terlama satu kali pengisian daya baterai akan bertahan saat Anda berada
    """)
    st.write("""19. three_g : Memiliki 3G atau tidak
    """)
    st.write("""20. touch screen : Memiliki layar sentuh atau tidak
    """)
    st.write("""21. wifi: Ada wifi atau tidak
    """)
    
    
    st.write("""Menggunakan Kolom (input) :

    precipitation
    tempmax * tempmin
    wind
    """)
    st.write("""Mobile Price Classification   (output) :
    
    1. 0    : tidak
    2. 1    : ya
    
    """)
    st.write("###### Aplikasi ini untuk : Mobile Price Classification (Klasifikasi ikan di Pasar) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/elmatiaaa/project")

with upload_data:
    # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     df = pd.read_csv(uploaded_file)
    #     st.write("Nama File Anda = ", uploaded_file.name)
    #     st.dataframe(df)
    df = pd.read_csv('https://raw.githubusercontent.com/elmatiaaa/Machine-Learning/main/test.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    df = df.drop(columns=['id', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'])
    #Mendefinisikan Varible X dan Y
    X = df[['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']]
    y = df['blue'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.blue).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],

        
    })

    st.write(labels)

    # st.subheader("""Normalisasi Data""")
    # st.write("""Rumus Normalisasi Data :""")
    # st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    # st.markdown("""
    # Dimana :
    # - X = data yang akan dinormalisasi atau data asli
    # - min = nilai minimum semua data asli
    # - max = nilai maksimum semua data asli
    # """)
    # df.weather.value_counts()
    # df = df.drop(columns=["date"])
    # #Mendefinisikan Varible X dan Y
    # X = df.drop(columns=['weather'])
    # y = df['weather'].values
    # df_min = X.min()
    # df_max = X.max()

    # #NORMALISASI NILAI X
    # scaler = MinMaxScaler()
    # #scaler.fit(features)
    # #scaler.transform(features)
    # scaled = scaler.fit_transform(X)
    # features_names = X.columns.copy()
    # #features_names.remove('label')
    # scaled_features = pd.DataFrame(scaled, columns=features_names)

    # #Save model normalisasi
    # from sklearn.utils.validation import joblib
    # norm = "normalisasi.save"
    # joblib.dump(scaled_features, norm) 


    # st.subheader('Hasil Normalisasi Data')
    # st.write(scaled_features)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        battery_power = st.number_input('masukkan Total energi yang dapat disimpan baterai dalam satu waktu diukur dalam mAh (battery power) : ')
        clock_speed = st.number_input('masukkan kecepatan di mana mikroprosesor mengeksekusi instruksi (clock speed) : ')
        fc = st.number_input('masukkan Kamera depan mega piksel (fc) : ')
        int_memory = st.number_input('masukkan Memori Internal dalam Gigabyte (int_memory) : ')
        m_dep = st.number_input('masukkan Kedalaman Seluler dalam cm (m_dep) : ')
        mobile_wt = st.number_input('masukkan Berat ponsel (mobile_wt) : ')
        n_cores = st.number_input('masukkan Jumlah inti prosesor (n_cores) : ')
        pc = st.number_input('masukkan Mega piksel Kamera Utama (pc) : ')
        px_height = st.number_input('masukkan Tinggi Resolusi Piksel (px_height) : ')
        px_width = st.number_input('masukkan lebar Resolusi Piksel (px_width) : ')
        ram = st.number_input('masukkan Memori Akses Acak dalam Megabita (ram) : ')
        sc_h = st.number_input('masukkan Tinggi layar ponsel dalam cm (sc_h) : ')
        sc_w = st.number_input('masukkan Lebar layar ponsel dalam cm (sc_w) : ')
        talk_time = st.number_input('masukkan waktu terlama satu kali pengisian daya baterai akan bertahan saat Anda berada (talk_time) : ')
      
        
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                battery_power, 
                clock_speed, 
                fc, 
                int_memory, 
                m_dep, 
                mobile_wt, 
                n_cores, 
                pc, 
                px_height, 
                px_width, 
                ram, 
                sc_h, 
                sc_w, 
                talk_time
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

               
            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)

