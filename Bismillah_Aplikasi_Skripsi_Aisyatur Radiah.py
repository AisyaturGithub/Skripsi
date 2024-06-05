import streamlit as st
from streamlit_option_menu import option_menu
import mysql.connector
import re
import pandas as pd
import numpy as np
import streamlit_wordcloud as wordcloud
from nltk.tag import CRFTagger
from nltk import pos_tag, word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define the calculate_precision function
def calculate_precision(relevant_documents, retrieved_documents):
    relevant_set = set(relevant_documents)
    retrieved_set = set(retrieved_documents)
    true_positives = len(relevant_set.intersection(retrieved_set))
    precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0
    return precision

#Fungsi Algoritma Levenshtein Distance
def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]

#Data yang tersimpan di database xampp, mysql
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    # database="skripsi"
    database="berita",
)

#Fungsi untuk menghubungkan data yang ada di database 
def correct_spelling_deletion(input_word):#koreksi kata berdasarkan kamus
   
    mycursor = mydb.cursor()#objek kursor pada koneksi database

    mycursor.execute("SELECT Unique_Words FROM data_kamus_pariwisata")##untuk mendapatkan kamus didalam database

    word_list = mycursor.fetchall()#untuk mengambil data dalam bentuk list
    mycursor.close()
    min_distance = float('inf')#
    corrected_word = input_word

    
    #memanggil correction spelling dari levenshtein distance
    # st.write(word_list[0][0])
    for word in word_list:
        # st.write(word[0])
        distance = levenshtein_distance(input_word, word[0])
        if distance < min_distance:
            min_distance = distance
            corrected_word = word[0]
    # st.write(corrected_word, min_distance)
    # if input_word == word:
        # st.write(input_word ,corrected_word, min_distance)

    return corrected_word

# Fungsi untuk koreksi ejaan
def correct_spelling(input_word, word_list):
    min_distance = float('inf')
    corrected_word = input_word

    for word in word_list:
        if input_word == word:
            return input_word
        
        for i in range(len(input_word)):
            edited_word = input_word[:i] + input_word[i+1:]
            distance = levenshtein_distance(edited_word, word)
            if distance < min_distance:
                min_distance = distance
                corrected_word = word
        
        for i in range(len(input_word) + 1):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                edited_word = input_word[:i] + char + input_word[i:]
                distance = levenshtein_distance(edited_word, word)
                if distance < min_distance:
                    min_distance = distance
                    corrected_word = word
        
        for i in range(len(input_word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                edited_word = input_word[:i] + char + input_word[i+1:]
                distance = levenshtein_distance(edited_word, word)
                if distance < min_distance:
                    min_distance = distance
                    corrected_word = word

    return corrected_word
#menghitung untuk mengetahui nilai presisi 
def correct_spelling(input_word, word_list):
    min_distance = float('inf')
    corrected_word = input_word

    for word in word_list:
        distance = levenshtein_distance(input_word, word)
        if distance < min_distance:
            min_distance = distance
            corrected_word = word

    return corrected_word

#untuk menghitung presisi
def calculate_precision(test_data, dictionary):
    true_positives = 0
    false_positives = 0

    for test_word, true_word in test_data:
        corrected_word = correct_spelling(test_word, dictionary)
        if corrected_word == true_word:
            true_positives += 1
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives)

    return precision

#  sama recall, ngebandingin data tes awal sama data koreksi atau dari kamusnya
# def calculate_precision_recall(test_data, dictionary):
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0

#     for test_word, true_word in test_data:
#         corrected_word = correct_spelling(test_word, dictionary)
#         if corrected_word == true_word:
#             true_positives += 1
#         else:
#             false_positives += 1

#     for true_word in dictionary:
#         if true_word not in [item[0] for item in test_data]:
#             false_negatives += 1

#     precision = true_positives / (true_positives + false_positives)
#     recall = true_positives / (true_positives + false_negatives)

#     return precision, recall

    # Fungsi untuk mengambil teks dari database
def fetch_text_from_database():
    cursor = mydb.cursor()
    cursor.execute("SELECT konten FROM data_berita_pariwisata__2_")  # Ganti 'tabel_berita' dengan nama tabel Anda
    result = cursor.fetchall()
    cursor.close()
    return [row[0] for row in result]
#     ## Visualisasi Data
#     # Menggabungkan semua teks dalam kolom Konten menjadi satu string
# all_text = ' '.join(data['Konten'])

# # Membuat objek WordCloud
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# # Menampilkan WordCloud
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()


    # Visualiasi Data
# def visualisasi(data):
#     # Membaca file CSV
#     # data = pd.read_csv('/xampp/htdocs/RISET_DIA/Data_Berita_Pariwisata_skripsi.csv')
#     # data
#     if len(data) == 0:
#         st.error("Tidak ada data yang sesuai dengan filter yang diterapkan.")
#         return
#     all_text = ' '.join(str(word) for word in df)
#     wordcloud = WordCloud(max_font_size=260, max_words=50, width=1000, height=1000, mode='RGBA', background_color='black').generate(all_text)#satu argumen dari data teks (yang ingin kita buat di awan kata)
#     plt.figure(figsize=(15,8))
#     plt.imshow(wordcloud, interpolation='bilinear') #menampilkan gambar yang lebih halus.
#     plt.axis("off")
#     plt.margins(x=0, y=0) 
#     st.pyplot(plt)
#     return data
# Memanggil fungsi visualisasi
#data = visualisasi('/xampp/htdocs/RISET_DIA/Data_Berita_Pariwisata_skripsi.csv')
# Fungsi untuk menghitung TF-IDF
def calculate_tfidf_from_database():
    corpus = fetch_text_from_database()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

# Fungsi untuk membuat inverted index
def create_inverted_index_from_database():
    corpus = fetch_text_from_database()
    inverted_index = {}
    for i, doc in enumerate(corpus):
        words = set(doc.split())
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = [i]
            else:
                inverted_index[word].append(i)
    return inverted_index

# # Contoh penggunaan
# tfidf_matrix, vectorizer = calculate_tfidf_from_database(link)
# inverted_index = create_inverted_index_from_database()
# Fungsi untuk melakukan pencarian berita berdasarkan cosine similarity
def search_news(query, tfidf_matrix, vectorizer, documents):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    results = []
    for i in related_docs_indices:
        results.append((documents[i], cosine_similarities[i]))
    return results

st.set_page_config(
    page_title="Spelling Correction System",
    page_icon='Logo_aplikasi_skripsi.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""<h1>Sistem Koreksi Ejaan Levenshtein Distance dan Part-of-Speech (POS) Tagging</h1>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        #st.write("""<h2 style = "text-align: center;"><a href=<a href="https://imgbb.com/"><img src="https://i.ibb.co/ckT90R7/Logo-apk-skripsi.png" alt="Logo-apk-skripsi" border="0" width="180" height="180"><br></h2>""",unsafe_allow_html=True)
        st.write("""<h2 style = "text-align: center;"><a href="https://imgbb.com/"><img src="https://i.ibb.co.com/ngnGwz9/logolatar-skripsi.png" alt="logolatar-skripsi" border="0" width="180" height="180"><br></h2>""",unsafe_allow_html=True), 
        ["Home", "Deskripsi","Dataset","Visualisasi", "Levenshtein Distance With Part-Of-Speech (POS) Tagging","Levenshtein Distance without Part-Of-Speech (POS) Tagging", "Testing","Referensi"], 
            icons=['house-door-fill', 'book-half','medium', 'bi bi-file-earmark-arrow-up-fill','bar-chart','bar-chart', 'gear', 'arrow-down-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#137ec2"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#87CEFA"}
            }
        )
if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://i.ibb.co.com/XXFMdQr/background-apk-skripsi.png" alt="background-apk-skripsi" border="0" width="700" height="370">
        </h3>""",unsafe_allow_html=True)

        st.write(""" """)
    
    
        st.write(""" """)

        st.write("""
        Spelling correction  sebuah fitur otomatis  digunakan 
        untuk proses pengoreksian kesalahan ejaan kata atau query.
        Kata kunci (query) yang diinputkan terjadi kesalahan ejaan 
        didalam query mesin pencarian (search engine), 
        salah satunya search engine bahasa Indonesia (SEBI).
        Mengakibatkan pemprosesan query tersebut mengembalikan 
        hasil yang tidak sesuai dengan kebutuhan informasi pengguna 
        atau disebut dengan error spelling correction. 
        Adanya kesalahan ejaan kata pada query ini dapat diatasi 
        menggunakan algoritma yaitu Levenshtein Distance.

        Akan tetapi algoritma Levenshtein Distance 
        tidak mampu mengoreksi kata bermakna banyak (ambiguitas kata). 
        Penambahan metode Part-of-Speech (POS) Tagging
        pada algoritma Levenshtein Distance diharapkan mampu mengetahui 
        pengaruh ambiguitas kata dalam error spelling correction.
        Serta seberapa pengaruh Part-of-Speech (POS) Tagging dalam 
        menghitung presision pada query pencarian artikel pariwisata.
        """)
#def visualisasi(data):
        #all_text_positive = ' '.join(str(word) for word in data['Ulasan'])
        #wordcloud = WordCloud(max_font_size=260, max_words=50, width=1000, height=1000, mode='RGBA', background_color='black').generate(all_text_positive)
        #plt.figure(figsize=(15,8))
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.margins(x=0, y=0) 
        #st.pyplot(plt)

if  selected == "Deskripsi":
    st.subheader("""Deskripsi Sistem""")
    st.write("""
         - Sistem ini adalah sistem koreksi ejaan pada search engine Bahasa Indonesia. 
         - Dataset yang digunakan yaitu artikel berita pariwisata.
         - Sistem ini menggunakan dua metode diantaranya sebagai berikut :
            1. Algoritma/ Metode Levenshtein Distance
            2. Metode Part-of-Speech (POS) Tagging
         - Sistem ini akan dilakukan dua operasi nilai presisi atau evaluasi sistem, yaitu 
           pada query yang diinputkan, dan information retrieval.
           Kedua proses tersebut akan diberi tag.
        """)

    st.subheader("""Tujuan Sistem""")
    st.write("""
        Tujuan dengan adanya sistem ini yaitu sebagai berikut :
        - Tujuan sistem ini untuk mengecek query salah dengan algoritma levenshtein distance. 
         - Penerapan metode levenshtein distance  untuk mengetahui nilai presisi dari koreksi ejaan 
           dan nilai presisi information retrieval.
         - Sistem ini terdapat penambahan metode untuk mengetahui ambiguitas kata (kata yang bermakna banyak)
         - Part-of-Speech (POS) Tagging diterapkan untuk pemberian tag pada setiap kata 
         - Penerapan koreksi ejaan dengan algoritma levenshtein distance dan penambahan 
           Part-of-Speech (POS) Tagging digunakan sebagai indikator dalam penelitian dalam 
           pengembangan sistem dengan topik search engine 
           dan Part-of-Speech (POS) Tagging, adanya ambiguitas kata (kata bermakna banyak).       
        """)

    st.subheader("""Manfaat Sistem""")
    st.write("""
        Manfaat sistem yaitu :
        - Sistem ini dapat dijadikan rujukan pada penelitian selanjutnya dengan topik spelling correction
        """)

elif selected == "Dataset":
        st.subheader("""Dataset Sistem""")
        st.write(""" Dataset sistem koreksi ejaan ini antara lain:
        <ol>
            <li>Dataset yang digunakan yaitu data artikel berita pariwisata</li>
            <li>Dataset diperoleh dari laman website detik.com melalui proses crawling</li>
            <li>Dataset terdiri dari judul, tanggal, link, konten</li>
            <li>Dataset diambil sampel untuk membuat data uji query benar dan query salah</li>
            <li>Bagian konten artikel berita pariwisata dilakukan proses pemberian tag (Part-of-Speech (POS)Tagging)</li>
            <li>Data berjumlah 332 Dokumen dengan 133.403 kata</li>
            <li>Dataset konten artikel yang dilakukan proses pembersihan/penghapus kata yang tidak penting atau terdapat tanda diberi tag berjumlah 132.689 kata dari 332 dokumen</li>
        </ol>
        """,unsafe_allow_html=True) 
        st.markdown("""<h2 style='text-align: center; color:grey;'> Dataset Artikel Berita Pariwisata </h1> """, unsafe_allow_html=True)
        df = pd.read_csv('https://raw.githubusercontent.com/aisyaturradiah/DATA-SKRIPSI/main/Data_Berita_Pariwisata.csv')
        c1, c2, c3 = st.columns([1,5,1])

        with c1:
            st.write("")

        with c2:
             df

        with c3:
            st.write("")
        st.markdown("""<h2 style='text-align: center; color:grey;'> Dataset Artikel Berita Pariwisata dengan Part-of-Speech (POS) Tagging </h1> """, unsafe_allow_html=True)
        df = pd.read_csv('https://raw.githubusercontent.com/aisyaturradiah/DATA-SKRIPSI/main/Data_Berita_Pariwisata_Postagging.csv')
        st.dataframe(df.style.format({'No': '{:.2f}','Artikel Pariwisata': '{:.2f}', 'Artikel Pariwisata': '{:.2f}'}))
       
        #st.subheader("""Dataset Artikel Berita Pariwisata Setelah dilakukan Part-of-Speech (POS) Tagging""")
        #st.markdown("""<h2 style='text-align: center; color:grey;'> Dataset Breast Cancer Prediction </h1> """, unsafe_allow_html=True)
        #df = pd.read_csv('https://raw.githubusercontent.com/aisyaturradiah/bismillah_skripsi/main/hasil_postagging_berita.csv')
        #c1, c2, c3 = st.columns([1,5,1])
        #df = pd.read_excel('nama_file.xlsx')
        #df = pd.read_csv('https://raw.githubusercontent.com/aisyaturradiah/bismillah_skripsi/main/hasil_postagging_berita.csv')
        #st.dataframe(df.style.format({'No': '{:.2f}','Artikel Berita': '{:.2f}'}))

# Define the calculate_precision function


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    # database="skripsi"
    database="berita",
)

def correct_spelling_deletion(input_word):#koreksi kata berdasarkan kamus
   
    mycursor = mydb.cursor()#objek kursor pada koneksi database

    mycursor.execute("SELECT Unique_Words FROM kamus_berita_pariwisata__2_")##untuk mendapatkan kamus didalam database

    word_list = mycursor.fetchall()#untuk mengambil data dalam bentuk list
    mycursor.close()
    min_distance = float('inf')#
    corrected_word = input_word
    
    #memanggil correction spelling dari levenshtein distance
    # st.write(word_list[0][0])
    for word in word_list:
        # st.write(word[0])
        distance = levenshtein_distance(input_word, word[0])
        if distance < min_distance:
            min_distance = distance
            corrected_word = word[0]
    # st.write(corrected_word, min_distance)
    # if input_word == word:
        # st.write(input_word ,corrected_word, min_distance)

    return corrected_word
#menghitung untuk mengetahui nilai presisi sama recall
def correct_spelling(input_word, word_list):
    min_distance = float('inf')
    corrected_word = input_word

    for word in word_list:
        distance = levenshtein_distance(input_word, word)
        if distance < min_distance:
            min_distance = distance
            corrected_word = word

    return corrected_word
#untuk menghitung presisi sama recall, ngebandingin data tes awal sama data koreksi atau dari kamusnya
def calculate_precision(test_data, dictionary):
    true_positives = 0
    false_positives = 0

    for test_word, true_word in test_data:
        corrected_word = correct_spelling(test_word, dictionary)
        
        if test_word == true_word:  # Kata asli sudah benar
            if corrected_word != test_word:
                false_positives += 1
        else:  # Kata asli salah
            if corrected_word == true_word:
                true_positives += 1
            else:
                false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    return precision

def calculate_precision(relevant_documents, retrieved_documents):
    # Convert numpy arrays to tuples to make them hashable
    relevant_set = set(map(tuple, relevant_documents))
    retrieved_set = set(map(tuple, retrieved_documents))
    true_positives = len(relevant_set.intersection(retrieved_set))
    precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0
    return precision
# def calculate_precision_recall(test_data, dictionary):
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0

#     for test_word, true_word in test_data:
#         corrected_word = correct_spelling(test_word, dictionary)
#         if corrected_word == true_word:
#             true_positives += 1
#         else:
#             false_positives += 1

#     for true_word in dictionary:
#         if true_word not in [item[0] for item in test_data]:
#             false_negatives += 1

#     precision = true_positives / (true_positives + false_positives)
#     recall = true_positives / (true_positives + false_negatives)

#     return precision, recall


if selected == "Visualisasi":
    st.write("""## Visualisasi""") #menampilkan judul halaman dataframe
    uploaded_files = st.file_uploader("Please choose a CSV file", type=['csv'])
    if uploaded_files is not None:
        data = pd.read_csv(uploaded_files, error_bad_lines=False)
        all_text = ' '.join(data['Konten'])

        # Try using default font or specify a font family
        font_path = r"C:\xampp\htdocs\RISET_DIA\AGENCYB.TTF" # Use default font or specify a font file path
           # Tetapkan family font sebagai 'AGENCYB'
        #font_family = 'AGENCYB'

            # Buat objek WordCloud dengan menyertakan parameter font_path dan font_family
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(all_text)
        
        # Display WordCloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        # Tampilkan WordCloud menggunakan streamlit
        st.image(wordcloud.to_array(), use_column_width=True)


if selected == "Levenshtein Distance With Part-Of-Speech (POS) Tagging":
    #with st.form("Implementation"):
    #st.write(levenshtein_distance("parawisata","pariwisata"))
    st.title("Skenario Levenshtein Distance & POS Tagging")
    col1, col2 = st.columns([3, 1])  # Adjust the column ratios as needed
    # Kunci unik untuk form masukan
    text_key = "koreksi_input"
    user_input = col1.text_input("Masukkan Query", key=text_key)
    col2.write("")
    col2.write("")
    # Menyediakan kunci unik untuk tombol
    button_key = "koreksi_button"
    periksa = col2.button("Koreksi Query", key=button_key)
    if periksa:
        #st.write(user_input)
        if user_input == "":
            st.warning("silahkan masukkan")
        else:
            user_input = user_input.lower()
            clean =[]
            for i in range (len(user_input)):
                clean_result  = re.sub("@[A-Za-z0-9_]+","", user_input[i]) #clenasing mention
                clean_result1 = re.sub("#[A-Za-z0-9_]+","", clean_result) #clenasing hashtag
                clean_result2 = re.sub(r'http\S+', '', clean_result1) #cleansing url link
                clean_result3 = re.sub("[^a-zA-Z ]+"," ", clean_result2) #cleansing character
                clean.append(clean_result3)

            tokenize = user_input.split()
            
            result = ""
            for input_word in tokenize:
                corrected_word = correct_spelling_deletion(input_word) 
                result += str(corrected_word) + " "   
                
            st.success(f"hasil correction: {result}") 
            data = CRFTagger()
            model_path ='all_indo_man_tag_corpus_model.crf.tagger'
            hasil1 = result.split()
            data.set_model_file(model_path)
            judul = data.tag_sents([result.split()])

            st.success(f"hasil Part-of-Speech (POS) Tagging:{judul}")
            # # Membaca data query yang benar dari file CSV
            # correct_queries_df = pd.read_csv("correct_queries.csv")  # Ganti "nama_file.csv" dengan nama file CSV yang sesuai
            # # Fungsi untuk menghitung precision
            # def calculate_precision(correct_query, corrected_query):
            #     return 1 if correct_query == corrected_query else 0

            #     # Menghitung precision untuk setiap data query
            # precisions = []
            # for correct_query in correct_queries_df["Correct Query"]:
            #     corrected_query = correct_spelling_deletion(correct_query)  # Ganti dengan fungsi koreksi ejaan yang Anda gunakan
            #     precision = calculate_precision(correct_query, corrected_query)
            #     precisions.append(precision)
            #                 # Menghitung rata-rata precision
            # average_precision = sum(precisions) / len(precisions)
            # # Menampilkan rata-rata precision
            # st.write(f"Rata-rata Precision: {average_precision}")

            # # Menampilkan hasil
            # print("Average Precision:", average_precision)
            
            # Hitung TF-IDF dan buat inverted index
#tfidf_matrix, vectorizer = calculate_tfidf_from_database()
#inverted_index = create_inverted_index_from_database()
            # Creating DataFrames
            df_input = pd.DataFrame(np.array(tokenize), columns=["Input"])
            df_result = pd.DataFrame(np.array(result.split()), columns=["Result"])
            concatenated_df = pd.concat([df_input, df_result], axis=1)
            hasil = np.array(concatenated_df)
            hasil_itung = []
            for index, row in concatenated_df.iterrows():
                input_value = row["Input"]
                result_value = row["Result"]
                precision = calculate_precision(input_value, result_value)
                hasil_itung.append({'Precision': precision})
            # Concatenating along columns (axis=1)
            concatenated_df = pd.concat([df_input, df_result], axis=1)
            hasil = np.array(concatenated_df)
            hasil_itung = []
            # for index, row in concatenated_df.iterrows():
            #     dictionary = np.array(result.split())
            #     data_test = np.array(row).reshape(1, -1)
            #     precision = calculate_precision(data_test, dictionary)
            #     # print(data_test)
            #     hasil_itung.append({
            #         'Precision': precision
            #     })
                # #precision, recall = calculate_precision_recall(data_test, dictionary)
                # hasil_itung.append({
                #     'Precision': precision,
                #     'Recall': recall
                # })

            df_hasil = pd.DataFrame(hasil_itung)
            # Calculate the average precision and recall
            #rata2recall = np.sum(df_hasil['Recall'].astype(float))  # Convert to float if not already
            rata2precision = np.mean(df_hasil['Precision'].astype(float))  # Convert to float if not already

            #st.info(f"Nilai Recall: {rata2recall}") 
            st.warning(f"Nilai Precision: {rata2precision}") 

            mycursor = mydb.cursor()
            #ini
            
            conditions = []
            for keyword in hasil1:
                conditions.append(f"judul LIKE '%{keyword}%' OR konten LIKE '%{keyword}%'")
            
    #Menampilkan judul dan konten
            
            mycursor.execute(f"SELECT * FROM data_berita_pariwisata__2_ WHERE " + " OR ".join(conditions) + ";")
            # mycursor.execute(f"SELECT * FROM data_berita_pariwisata__2_ WHERE judul LIKE '%{result}%' or konten LIKE '%{result}%'")
            judul_list = mycursor.fetchall()
            #st.write(f"judul_list")
            mycursor.close()
            if len(judul_list) == 0:
                st.warning("Tidak Ada Berita")
            else:
                # Ambil konten artikel dari judul_list
                corpus = [judul[4] for judul in judul_list]
                # Hitung TF-IDF dan buat inverted index
                # tfidf_matrix, vectorizer = calculate_tfidf_from_database(link)
                # corpus = fetch_text_from_database(link)
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(corpus)
                inverted_index = create_inverted_index_from_database()

                # Ambil input query dari pengguna
                user_query = result  # Menggunakan hasil koreksi ejaan sebagai query

                # Jika ada query dari pengguna, cari artikel yang sesuai
                if user_query:
                    # Hitung vektor representasi query
                    query_vec = vectorizer.transform([user_query])

                    # Cari dokumen yang paling mirip dengan query menggunakan cosine similarity
                    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
                    related_docs_indices = cosine_similarities.argsort()[::-1]
                if len(judul_list) > 0:
                    def hitung_presisi(judul_tampil, judul_relevan):
                        # Menghitung jumlah judul yang relevan yang ditampilkan
                        total_judul_relevan_ditampilkan = 0
                        for judul in judul_tampil:
                            # Cek apakah semua kata kunci relevan ada dalam judul
                            if all(kata in judul.lower() for kata in judul_relevan):
                                total_judul_relevan_ditampilkan += 1

                        # Menghitung presisi
                        if len(judul_tampil) > 0:
                            presisi = total_judul_relevan_ditampilkan / len(judul_tampil)
                        else:
                            presisi = 0                         

                        return presisi
    # Inisialisasi CRFTagger dilakukan di luar loop karena kita hanya perlu melakukannya sekali
                    data = CRFTagger()
                    model_path = 'all_indo_man_tag_corpus_model.crf.tagger'
                    data.set_model_file(model_path)
                    displayed_documents = []
                    judul_tersimpan = []  
                    for i in related_docs_indices[:min(5, len(judul_list))]:
                        if i <= len(judul_list):   
                            # st.header(f"Judul: {judul_list[i][1]}")#menampilkan judul artikel
                            judul_tersimpan.append(judul_list[i][1])
                            #st.write(f"Konten: {judul_list[i][4]}")#menampilkan artikel
                            st.markdown(
                                f'<h3><a style="color: #778899;" href="{judul_list[i][3]}">{judul_list[i][1]}</a></h3>', unsafe_allow_html=True)
                            st.write(f"{judul_list[i][2]}") #menampilkan tanggal dan waktu
                            st.markdown(judul_list[i][4][:200])
                            #judul_tagged = judul_tagged[0]  # Kembalikan ke bentuk yang diharapkan
                            #judul_tagged = ' '.join([f'{word}/{tag}' for word, tag in judul_tagged])  # Gabungkan kembali kata-kata dengan tag
                            # POS Tagging pada judul artikel
                            judul_tagged = judul_list[i][1].split()
                            judul_tagged = data.tag_sents([judul_tagged])
                            #judul_tagged ='test'

                            st.success(f"Hasil Part-of-Speech (POS) Tagging untuk judul artikel: {judul_tagged}")
                            #displayed_documents.append(judul_list[i][1])
                            # POS Tagging pada judul artikel
                            # tanggal_tagged = judul_list[i][2].split()
                            # tanggal_tagged = data.tag_sents([tanggal_tagged])
                            #judul_tagged ='test'
                            # print(tanggal_tagged)
                            # st.success(f"Hasil Part-of-Speech (POS) Tagging untuk judul artikel: {tanggal_tagged}")
                            #displayed_documents.append(judul_list[i][2])
                            # POS Tagging pada konten artikel
                            konten_tagged = judul_list[i][4].split()
                            konten_tagged = data.tag_sents([konten_tagged])
                            
                            # st.success(f"Hasil Part-of-Speech (POS) Tagging untuk konten artikel: {konten_tagged}")
                            displayed_documents.append(judul_list[i][4])
                    
                    # st.write("Presisi:", presisi)
                    
                    st.session_state.judul_tersimpan = judul_tersimpan       
                    presisi1 = hitung_presisi(judul_tersimpan, hasil1)

                    st.warning(f"Nilai Presisi: {presisi1}")
                    # st.info(f"Nilai Precision: {rata2precision}") 
                    # relevant_documents = [judul[2] for judul in judul_list]
                    # precision_articles = calculate_precision(relevant_documents, displayed_documents)
                    # st.info(f"Presisi dari artikel yang ditampilkan: {precision_articles}")
        #  else:        # Assuming you have a list of relevant documents for the query
                    # relevant_documents = [documents[4] for documents in judul_list if documents[4] in displayed_documents]
                    # test_data_relevant = [(doc, doc) for doc in relevant_documents]
                    # precision = calculate_precision(test_data_relevant, dictionary)
                    # st.info(f"Presisi dari artikel yang ditampilkan: {precision}")   
        #     st.write("Indeks dokumen melebihi jumlah dokumen yang tersedia.")
                                # Concatenating along columns (axis=1)
                                # concatenated_df = pd.concat([df_input, df_result], axis=1)
                                # hasil_itung = []
                                # for index, row in concatenated_df.iterrows():
                                #     dictionary = np.array(result.split())
                                #     data_test = np.array(row).reshape(1, -1)
                                #     precision = calculate_precision(data_test, dictionary)
                                #     hasil_itung.append({
                                #         'Precision': precision
                                #     })

                                # # Calculate the average precision
                                # rata2precision = np.mean(df_hasil['Precision'].astype(float))

                                # st.info(f"Nilai Precision: {rata2precision}")


            # # Calculate the average precision and recall
            # #rata2recall = np.sum(df_hasil['Recall'].astype(float))  # Convert to float if not already
            # rata2precision = np.mean(df_hasil['Precision'].astype(float))  # Convert to float if not already

            # #st.info(f"Nilai Recall: {rata2recall}") 
            # st.info(f"Nilai Precision: {rata2precision}") 


elif selected == "Levenshtein Distance without Part-Of-Speech (POS) Tagging":
        
        st.title("Skenario Levenshtein Distance Tanpa POS Tagging ")
        col1, col2 = st.columns([3, 1])  # Adjust the column ratios as needed
        # Kunci unik untuk form masukan
        text_key = "koreksi_input"
        user_input = col1.text_input("Masukkan Query", key=text_key)
        col2.write("")
        col2.write("")
        # Menyediakan kunci unik untuk tombol
        button_key = "koreksi_button"
        periksa = col2.button("Koreksi Query", key=button_key)
        if periksa:
            #st.write(user_input)
            if user_input == "":
                st.warning("silahkan masukkan")
            else:
                user_input = user_input.lower()
                clean =[]
                for i in range (len(user_input)):
                    clean_result  = re.sub("@[A-Za-z0-9_]+","", user_input[i]) #clenasing mention
                    clean_result1 = re.sub("#[A-Za-z0-9_]+","", clean_result) #clenasing hashtag
                    clean_result2 = re.sub(r'http\S+', '', clean_result1) #cleansing url link
                    clean_result3 = re.sub("[^a-zA-Z ]+"," ", clean_result2) #cleansing character
                    clean.append(clean_result3)

                tokenize = user_input.split()
                
                result = ""
                for input_word in tokenize:
                    corrected_word = correct_spelling_deletion(input_word) 
                    result += str(corrected_word) + " "   
                   
                st.success(f"hasil correction: {result}")
                hasil1 = result.split()
                # Split hasil koreksi ejaan menjadi kata-kata individu
             
                # Membaca data query yang benar dari file CSV
                # correct_queries_df = pd.read_csv("correct_queries.csv")  # Ganti "nama_file.csv" dengan nama file CSV yang sesuai
                # # Fungsi untuk menghitung precision
                # def calculate_precision(correct_query, corrected_query):
                #     return 1 if correct_query == corrected_query else 0

                #     # Menghitung precision untuk setiap data query
                # precisions = []
                # for correct_query in correct_queries_df["Correct Query"]:
                #     corrected_query = correct_spelling_deletion(correct_query)  # Ganti dengan fungsi koreksi ejaan yang Anda gunakan
                #     precision = calculate_precision(correct_query, corrected_query)
                #     precisions.append(precision)
                #                 # Menghitung rata-rata precision
                # average_precision = sum(precisions) / len(precisions)
                # # Menampilkan rata-rata precision
                # st.write(f"Rata-rata Precision: {average_precision}")

                # # Menampilkan hasil
                # print("Average Precision:", average_precision)
                
                # Creating DataFrames
                df_input = pd.DataFrame(np.array(tokenize), columns=["Input"])
                df_result = pd.DataFrame(np.array(result.split()), columns=["Result"])

                # Concatenating along columns (axis=1)
                concatenated_df = pd.concat([df_input, df_result], axis=1)
                hasil = np.array(concatenated_df)
                hasil_itung = []
                for index, row in concatenated_df.iterrows():
                    input_value = row["Input"]
                    result_value = row["Result"]
                    precision = calculate_precision(input_value, result_value)
                    hasil_itung.append({'Precision': precision})
                # for index, row in concatenated_df.iterrows():
                #     dictionary = np.array(result.split())
                #     data_test = np.array(row).reshape(1, -1)
                #     precision = calculate_precision(data_test, dictionary)
                #     # print(data_test)
                #     hasil_itung.append({
                #         'Precision': precision
                #     })
                df_hasil = pd.DataFrame(hasil_itung)

                # Calculate the average precision and recall
                #rata2recall = np.sum(df_hasil['Recall'].astype(float))  # Convert to float if not already
                rata2precision = np.mean(df_hasil['Precision'].astype(float))  # Convert to float if not already

                #st.info(f"Nilai Recall: {rata2recall}") 
                st.info(f"Nilai Precision: {rata2precision}") 

                mycursor = mydb.cursor()
                conditions = []
                for keyword in hasil1:
                    conditions.append(f"judul LIKE '%{keyword}%' OR konten LIKE '%{keyword}%'")
            
                mycursor.execute(f"SELECT * FROM data_berita_pariwisata__2_ WHERE " + " OR ".join(conditions) + ";")
                #Menampilkan Judul dan Konten
                #mycursor.execute(f"SELECT * FROM data_berita_pariwisata__2_ WHERE judul LIKE '%{result}%' or konten LIKE '%{result}%'")
                judul_list = mycursor.fetchall()
                #st.write(f"judul_list")
                mycursor.close()
                if len(judul_list) == 0:
                    st.warning("Tidak Ada Berita")
                else:
                # Ambil konten artikel dari judul_list
                    corpus = [judul[4] for judul in judul_list]
                    
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform(corpus)
                    inverted_index = create_inverted_index_from_database()
                   
                    # Ambil input query dari pengguna
                    user_query = result  # Menggunakan hasil koreksi ejaan sebagai query

                # Jika ada query dari pengguna, cari artikel yang sesuai
                    if  user_query:
                    # Hitung vektor representasi query
                        query_vec = vectorizer.transform([user_query])

                        # Cari dokumen yang paling mirip dengan query menggunakan cosine similarity
                        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
                        related_docs_indices = cosine_similarities.argsort()[::1]
                    
                    if len(judul_list) > 0:
                        def hitung_presisi(judul_tampil, judul_relevan):
                            # Menghitung jumlah judul yang relevan yang ditampilkan
                            total_judul_relevan_ditampilkan = 0
                            for judul in judul_tampil:
                                # Cek apakah semua kata kunci relevan ada dalam judul
                                if all(kata in judul.lower() for kata in judul_relevan):
                                    total_judul_relevan_ditampilkan += 1

                            # Menghitung presisi
                            if len(judul_tampil) > 0:
                                presisi = total_judul_relevan_ditampilkan / len(judul_tampil)
                            else:
                                presisi = 0                         

                            return presisi
                    displayed_documents = []
                    judul_tersimpan = []
                    if len(related_docs_indices) > 0:
                        num_articles = min(5, len(related_docs_indices))
                        matched_articles = 0  # Variabel untuk menghitung jumlah artikel yang sesuai dengan query
                        for i in range(num_articles):
                            index = related_docs_indices[i]
                            if i < len(judul_list): 
                                st.header(f"Judul: {judul_list[i][1]}")#menampilkan judul artikel
                                judul_tersimpan.append(judul_list[i][1])
                                st.subheader(f"Tanggal: {judul_list[i][2]}") #menampilkan tanggal dan waktu
                                st.write(f"link: {judul_list[i][3]}")
                                #st.write(f"Konten: {judul_list[i][4]}")#menampilkan artikel
                                st.markdown(
                                     f'<h3><a style="color: #778899;" href="{judul_list[i][3]}">{judul_list[i][1]}</a></h3>', unsafe_allow_html=True)
                                st.write(f"{judul_list[i][2]}") #menampilkan tanggal dan waktu
                                st.markdown(judul_list[i][4][:200])
                                # judul  = judul_list[i][1].split()
                                # konten = judul_list[i][4].split()

                                displayed_documents.append(judul_list[i][4])
                                
                    st.session_state.judul_tersimpan = judul_tersimpan       
                    presisi1 = hitung_presisi(judul_tersimpan, hasil1)

                    st.warning(f"Nilai Presisi: {presisi1}")
                                # # menggunakan query yang data 100
                                # concatenated_df = pd.concat([df_input, df_result], axis=1)
                                # hasil_itung = []
                                # for index, row in concatenated_df.iterrows():
                                #     input_value = row["Input"]
                                #     result_value = row["Result"]
                                #     precision = calculate_precision(input_value, result_value)
                                #     hasil_itung.append({'Precision': precision})
                                # for index, row in concatenated_df.iterrows():
                                #     dictionary = np.array(result.split())
                                #     data_test = np.array(row).reshape(1, -1)
                                #     precision = calculate_precision(data_test, dictionary)
                                #     hasil_itung.append({
                                #         'Precision': precision
                                #     })

                                # Calculate the average precision
                                # rata2precision = np.mean(df_hasil['Precision'].astype(float))

                                # st.info(f"Nilai Precision: {rata2precision}")
                                # for index, row in concatenated_df.iterrows():
                                #     input_value = row["Input"]
                                #     result_value = row["Result"]
                                #     precision = calculate_precision(input_value, result_value)
                                #     hasil_itung.append({'Precision': precision})
                                # for index, row in concatenated_df.iterrows():
                                #     dictionary = np.array(result.split())
                                #     data_test = np.array(row).reshape(1, -1)
                                #     precision = calculate_precision(data_test, dictionary)
                                #     hasil_itung.append({
                                #         'Precision': precision
                                #     })


                                # Calculate the average precision
                                # rata2precision = np.mean(df_hasil['Precision'].astype(float))

                                # st.info(f"Nilai Precision: {rata2precision}")

# option = st.sidebar.selectbox(
#     'Silakan pilih:',
#     ('Testing','Testing POS Tagging','Testing Tanpa POS Tagging')
# )           
# if option == 'Testing' or option == '':
#     st.write("""# Proses Testing Data Query yang Berjumlah 100 Data""") #menampilkan halaman utama      



elif selected== "Testing": 
    option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('💯Testing  Tanpa POS Tagging','💯Testing POS Tagging')
)           
    if option == '💯Testing  Tanpa POS Tagging' or option == '':
        #st.write("""# Proses Testing Data Query yang Berjumlah 100 Data""") #menampilkan halaman utama      
        
            #Fungsi untuk menghubungkan data yang ada di database 
        def correct_spelling_deletion(input_word):#koreksi kata berdasarkan kamus
        
            mycursor = mydb.cursor()#objek kursor pada koneksi database

            mycursor.execute("SELECT Unique_Words FROM data_kamus_pariwisata")##untuk mendapatkan kamus didalam database

            word_list = mycursor.fetchall()#untuk mengambil data dalam bentuk list
            mycursor.close()
            min_distance = float('inf')#
            corrected_word = input_word
#memanggil correction spelling dari levenshtein distance
            # st.write(word_list[0][0])
            for word in word_list:
                # st.write(word[0])
                distance = levenshtein_distance(input_word, word[0])
                if distance < min_distance:
                    min_distance = distance
                    corrected_word = word[0]
            # st.write(corrected_word, min_distance)
            # if input_word == word:
                # st.write(input_word ,corrected_word, min_distance)
            return corrected_word

                    #menghitung untuk mengetahui nilai presisi 
        def correct_spelling(input_word, word_list):
            min_distance = float('inf')
            corrected_word = input_word

            for word in word_list:
                distance = levenshtein_distance(input_word, word)
                if distance < min_distance:
                    min_distance = distance
                    corrected_word = word

            return corrected_word

                    #untuk menghitung presisi
        # Fungsi untuk menghitung presisi
        def calculate_precision(data_test, dictionary):
            benar = 0
            salah = 0

            for test_word, true_word in data_test:
                corrected_word = correct_spelling(test_word, dictionary)
                if corrected_word == true_word:
                    benar += 1
                else:
                    salah += 1

            total_kata = benar + salah
            if total_kata == 0:
                return 0.0
            else:
                precision = benar / total_kata
                return precision
        # def calculate_precision(test_data, dictionary):
        #     true_positives = 0
        #     false_positives = 0

        #     for test_word, true_word in test_data:
        #         corrected_word = correct_spelling(test_word, dictionary)
        #         if corrected_word == true_word:
        #             true_positives += 1
        #         else:
        #             false_positives += 1

        #     precision = true_positives / (true_positives + false_positives)

        #     return precision

        st.title("Testing Data Test")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file: 
                data = pd.read_csv(uploaded_file)
                st.subheader("Dataset Pengujian")
                st.dataframe(data)
                def bersihkan_teks(teks):
                    # Menghapus huruf besar
                    teks = teks.lower()

                    # Menghapus karakter khusus dan tanda baca
                    teks = re.sub(r'[^a-zA-Z0-9\s]', '', teks)

                    # Menghapus tautan (link)
                    teks = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', teks)

                    return teks

                # #untuk melakukan komputasi dari presisi dari correction 
                def komputasi(row):
                    clean = bersihkan_teks(row[0])
                    tokenize = clean.split()
                    result = ""
                    for input_word in tokenize:
                        corrected_word = correct_spelling_deletion(input_word) 
                        result += str(corrected_word) + " " 
                        
                    df_input = pd.DataFrame(np.array(tokenize), columns=["Input"])
                    df_result = pd.DataFrame(np.array(result.split()), columns=["Result"])

                    concatenated_df = pd.concat([df_input, df_result], axis=1)
                    hasil = np.array(concatenated_df)
                    hasil_itung = []
                    for index, row in concatenated_df.iterrows():
                        dictionary = np.array(result.split())
                        data_test = np.array(row).reshape(1, -1)
                        precision = calculate_precision(data_test, dictionary)
                        hasil_itung.append({
                            'Precision': precision,
                        })

                    df_hasil = pd.DataFrame(hasil_itung)
                    rata2precision = np.mean(df_hasil['Precision'].astype(float))  # Convert to float if not already
                    return result, rata2precision
                             
                hasil_koreksi = []
                precision = []
                for index, row in data.iterrows():
                    result,rata2precision = komputasi(row)
                    hasil_koreksi.append(result)
                    precision.append(rata2precision)

                # menampilkan hasil koreksi kata pada data uji
                st.subheader("Hasil Koreksi kata pada Data test")
                hasil_koreksi_kata = pd.DataFrame(hasil_koreksi, columns = ['Hasil Koreksi Kata'])
                #hasil_koreksi_kata.to_excel("/xampp/htdocs/RISET_DIA/hasil_koreksi_tanpaPOS.xlsx")
                hasil_perbandingan = pd.concat([data,hasil_koreksi_kata],axis=1)
                st.dataframe(hasil_perbandingan)

                # Membuat DataFrame hasil
                df_hasil = pd.DataFrame({
                    'Hasil Koreksi': hasil_koreksi,
                    'Presisi': precision
                })
                st.write(hasil_koreksi)
                st.subheader("Hasil Koreksi")
                st.dataframe(df_hasil)

                rata2_precision = np.mean(precision)
                st.write("Rata-rata Presisi:", rata2_precision)
                # # menampilkan hasil koreksi kata pada data uji
                st.subheader("Hasil Koreksi kata pada Data test")
                hasil_koreksi_kata = pd.DataFrame(hasil_koreksi, columns = ['Hasil Koreksi Kata'])
                #hasil_koreksi_kata.to_excel("/xampp/htdocs/RISET_DIA/hasil_koreksi_tanpaPOS.xlsx")
                hasil_perbandingan = pd.concat([data,hasil_koreksi_kata],axis=1)
                st.dataframe(hasil_perbandingan)

                # Menampilkan hasil precision
                st.subheader("Hasil Precision")
                precision_data = {
                    "Query": [f"Query {i+1}" for i in range(len(precision))],
                    "Precision": precision
                }
                hasil_precision = pd.DataFrame(precision_data)
                st.dataframe(hasil_precision)
                #mendowload hasil perhitungan presisi
                #hasil_precision.to_excel("/xampp/htdocs/RISET_DIA/50data_Tanpa Pos.xlsx")
                st.line_chart(hasil_precision.set_index('Query')['Precision'])  # Menampilkan line chart dari nilai precision
                rata2precision = np.mean(hasil_precision['Precision'])
                st.info(f"Rata-rata Precision: {rata2precision:.2f}")

                # Menampilkan grafik presisi dengan nama query
                st.subheader("Grafik Precision dengan Nama Query")
                st.line_chart(hasil_precision.set_index('Query'))
                st.write(f"Total Precision: {hasil_precision['Precision'].sum():.2f}")

                # Hasil recall dan precision 
                st.subheader("Hasil Precision")
                st.line_chart(hasil_precision['Precision'])  # Display precision without Query as x-axis  # Menampilkan line chart dari nilai precision
                # hasil_recall = pd.DataFrame(recall)
                # rata2recall = np.mean(hasil_recall[0])
                #st.info(f"Recall: {rata2recall}") 

                hasil_precision = pd.DataFrame(precision)
                rata2precision = np.mean(hasil_precision[0])
                st.info(f"Precision: {rata2precision:.2%}")

                st.subheader("Hasil Precision")
                hasil_precision = pd.DataFrame(precision, columns=['Precision'])
                #st.line_chart(hasil_precision.set_index('Query_Salah'))
                st.line_chart(hasil_precision)  # Menampilkan line chart dari nilai precision
                rata2precision = np.mean(hasil_precision['Precision'])
                st.info(f"Precision: {rata2precision}")
        else:
                st.warning('Masukkan Data terlebih dahulu')
        
       ###
        def correct_spelling_deletion(input_word):#koreksi kata berdasarkan kamus
        
            mycursor = mydb.cursor()#objek kursor pada koneksi database

            mycursor.execute("SELECT Unique_Words FROM data_kamus_pariwisata")##untuk mendapatkan kamus didalam database

            word_list = mycursor.fetchall()#untuk mengambil data dalam bentuk list
            mycursor.close()
            min_distance = float('inf')#
            corrected_word = input_word
#memanggil correction spelling dari levenshtein distance
            # st.write(word_list[0][0])
            for word in word_list:
                # st.write(word[0])
                distance = levenshtein_distance(input_word, word[0])
                if distance < min_distance:
                    min_distance = distance
                    corrected_word = word[0]
            # st.write(corrected_word, min_distance)
            # if input_word == word:
                # st.write(input_word ,corrected_word, min_distance)
            return corrected_word

                    #menghitung untuk mengetahui nilai presisi 
        def correct_spelling(input_word, word_list):
            min_distance = float('inf')
            corrected_word = input_word

            for word in word_list:
                distance = levenshtein_distance(input_word, word)
                if distance < min_distance:
                    min_distance = distance
                    corrected_word = word

            return corrected_word


    elif option == '💯Testing POS Tagging':
        st.title("Testing Data Test")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file: 
                dataset = pd.read_csv(uploaded_file)
                st.subheader("Dataset Pengujian")
                st.dataframe(dataset)
            # Fetch the word list once
                def fetch_word_list():
                    mycursor = mydb.cursor()
                    mycursor.execute("SELECT Unique_Words FROM data_kamus_pariwisata")
                    word_list = [word[0] for word in mycursor.fetchall()]
                    mycursor.close()
                    return word_list
        # #dataset = pd.read_csv('data_testt.csv')
        # st.subheader("Dataset Pengujian")
        # #st.dataframe(dataset)
                def bersihkan_teks(teks):
                    # Menghapus huruf besar
                    teks = teks.lower() 

                    # Menghapus karakter khusus dan tanda baca
                    teks = re.sub(r'[^a-zA-Z0-9\s]', '', teks)

                    # Menghapus tautan (link)
                    teks = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', teks)

                    return teks
            
                def calculate_precision(data_test, dictionary):
                    benar = 0
                    salah = 0
                    for test_word, true_word in data_test:
                        corrected_word = correct_spelling(test_word, dictionary)
                        if corrected_word == true_word:
                            benar += 1
                        else:
                            salah += 1
                    total_kata = benar + salah
                    return benar / total_kata if total_kata > 0 else 0.0
            # # Function to calculate precision
            # def calculate_precision(true_tags, predicted_tags):
            #     correct = sum(t1 == t2 for t1, t2 in zip(true_tags, predicted_tags))
            #     total = len(true_tags)
            #     return correct / total if total > 0 else 0
            #Function to compute precision and perform POS tagging
                def komputasi(row, tagger,word_list):
                    clean = bersihkan_teks(row[0])
                    tokenize = clean.split()
                    result = ""
                    precision = [] 
                    for input_word in tokenize:
                        corrected_word = correct_spelling(input_word,word_list) 
                        result += str(corrected_word) + " "
                        precision.append((input_word, corrected_word))

                    df_input = pd.DataFrame(np.array(tokenize), columns=["Input"])
                    df_result = pd.DataFrame(np.array(result.split()), columns=["Result"])

                    # Concatenating along columns (axis=1)
                    concatenated_df = pd.concat([df_input, df_result], axis=1)
                    hasil = np.array(concatenated_df)
                    hasil_itung = []
                    for index, row in concatenated_df.iterrows():
                        dictionary = np.array(result.split())
                        data_test = np.array(row).reshape(1, -1)
                        precision = calculate_precision(data_test,dictionary)
                        hasil_itung.append({
                            'Precision': precision,
                        })

                    df_hasil = pd.DataFrame(hasil_itung)

                        # Calculate the average precision and recall
                        #rata2recall = np.sum(df_hasil['Recall'].astype(float))  # Convert to float if not already
                    rata2precision = np.mean(df_hasil['Precision'].astype(float))  # Convert to float if not already

                    # POS Tagging
                    tagger.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
                    pos_tags = tagger.tag_sents([result.split()])

                    return result, rata2precision, pos_tags
        # Initialize the CRFTagger
                tagger = CRFTagger()
                #untuk melakukan perhitungan koreksi kata presisi 
                hasil_koreksi = []
                #recall = []
                precision = []
                pos_tagging_results = []
                word_list = fetch_word_list()
                for index, row in dataset.iterrows():
                    result,rata2precision, pos_tags = komputasi(row, tagger,word_list)
                    hasil_koreksi.append(result)
                    #recall.append(rata2recall)
                    precision.append(rata2precision)
                    pos_tagging_results.append(pos_tags)

                # menampilkan hasil koreksi kata pada data uji
                st.subheader("Hasil Koreksi kata pada Data test")
                hasil_koreksi_kata = pd.DataFrame(hasil_koreksi, columns = ['Hasil Koreksi Kata'])
                #hasil_koreksi_kata.to_excel("/xampp/htdocs/RISET_DIA/hasil_koreksi_POS.xlsx")
                hasil_perbandingan = pd.concat([dataset,hasil_koreksi_kata],axis=1)
                st.dataframe(hasil_perbandingan)

                # Menampilkan hasil precision
                st.subheader("Hasil Precision")
                precision_dataset = {
                    "Query": [f"Query {i+1}" for i in range(len(precision))],
                    "Precision": precision
                }
                hasil_precision = pd.DataFrame(precision_dataset)
                st.dataframe(hasil_precision)
                # Display POS tagging results
                st.subheader("Hasil Part-of-Speech (POS) Tagging")
                pos_tagging_df = pd.DataFrame(pos_tagging_results, columns=['POS Tagging'])
                st.dataframe(pos_tagging_df)
                #mendowload hasil perhitungan presisi
                #hasil_precision.to_excel("/xampp/htdocs/RISET_DIA/50data_hasil_POS.xlsx")
                st.line_chart(hasil_precision.set_index('Query')['Precision'])  # Menampilkan line chart dari nilai precision
                rata2precision = np.mean(hasil_precision['Precision'])
                st.info(f"Rata-rata Precision: {rata2precision:.2f}")


                # Menampilkan grafik presisi dengan nama query
                st.subheader("Grafik Precision dengan Nama Query")
                st.line_chart(hasil_precision.set_index('Query'))
                st.write(f"Total Precision: {hasil_precision['Precision'].sum():.2f}")

                # Hasil recall dan precision 
                st.subheader("Hasil Precision")
                #st.line_chart(hasil_precision['Precision'])  # Display precision without Query as x-axis  # Menampilkan line chart dari nilai precision
                # hasil_recall = pd.DataFrame(recall)
                # rata2recall = np.mean(hasil_recall[0])
                # st.info(f"Recall: {rata2recall}") 

                hasil_precision = pd.DataFrame(precision)
                rata2precision = np.mean(hasil_precision[0])
                st.info(f"Precision: {rata2precision:.2%}")

                


        

                # import numpy as np
            # import matplotlib.pyplot as plt

            # np.random.seed(0)

            # n_bins = 10
            # x = np.random.randn(1000, 3)

            # # Memilih kolom pertama dari array x untuk digunakan dalam histogram
            # data_set = x[:, 0]
            # # Membuat plot untuk nilai presisi
            # fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 10))  # Dua sumbu, satu untuk nilai presisi dan satu untuk histogram

            # # # Membuat plot untuk nilai presisi
            # # fig, ax = plt.subplots()
            # # ax.bar(['Precision'], [rata2precision], color='skyblue')
            # # ax.set_ylabel('Nilai')
            # # ax.set_title('Grafik Nilai Presisi')

            # #colors = ['red', 'yellow', 'blue']
            # # Menambahkan label nilai pada bar
            # #ax.text(0, rata2precision + 0.01, f'{rata2precision:.2f}', ha='center')
            # # Membuat subplot untuk data pengujian
            # #fig, ax0 = plt.subplots()
            # # Histogram
            # ax0.hist(x[:, 0], bins=n_bins, density=True, histtype='bar', color='blue', label='Dataset 1')
            # # ax0.hist(x[:, 1], bins=n_bins, density=True, histtype='bar', color='yellow', alpha=0.7, label='Dataset 2')
            # # ax0.hist(x[:, 2], bins=n_bins, density=True, histtype='bar', color='blue', alpha=0.7, label='Dataset 3')
            # ax0.legend(prop={'size': 10})
            # ax0.set_title('Histogram Data')
            # ax0.set_xlabel('Nilai')
            # ax0.set_ylabel('Frekuensi')
            # # Menampilkan nilai presisi
            # rata2precision = np.mean(precision)  # Rata-rata nilai presisi
            # ax0.text(0.05, 0.95, f'Precision: {rata2precision:.2f}', transform=ax0.transAxes, fontsize=12,
            #         verticalalignment='top')

            # # Plot nilai presisi
            # ax1.bar(['Precision'], [rata2precision], color='skyblue')
            # ax1.set_ylabel('Nilai')
            # ax1.set_title('Grafik Nilai Presisi')

            # # Menambahkan label nilai pada bar
            # ax1.text(0, rata2precision + 0.01, f'{rata2precision:.2f}', ha='center')


            # fig.tight_layout()

            # # Menampilkan plot menggunakan st.pyplot()
            # st.pyplot(fig)
            # ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
            # ax1.set_title('stacked bar')

            # ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
            # ax2.set_title('stack step (unfilled)')

            # # Make a multiple-histogram of data-sets with different length.
            # x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
            # ax3.hist(x_multi, n_bins, histtype='bar')
            # ax3.set_title('different sample sizes')

            # fig.tight_layout()
            
            # # Menampilkan plot menggunakan st.pyplot()
            # st.pyplot(fig)

        #     import numpy as np
        #     import matplotlib.pyplot as plt

        #     # Generate random data for demonstration purposes
        #     # You can replace this with your actual precision data
        #     np.random.seed(0)
        #     precision_data = np.random.rand(100)

        #     # Create histogram
        #     plt.hist(precision_data, bins=10, color='blue', alpha=0.7)

        #     # Add labels and title
        #     plt.xlabel('Precision')
        #     plt.ylabel('Frequency')
        #     plt.title('Histogram of Precision')

        #   # Show plot
        #     st.pyplot(plt) # Menampilkan plot menggunakan st.pyplot()
                    