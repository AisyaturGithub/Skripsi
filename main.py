import streamlit as st
import mysql.connector
import re
import pandas as pd
import numpy as np
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
    false_negatives = 0

    for test_word, true_word in test_data:
        corrected_word = correct_spelling(test_word, dictionary)
        if corrected_word == true_word:
            true_positives += 1
        else:
            false_positives += 1

    for true_word in dictionary:
        if true_word not in [item[0] for item in test_data]:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    # recall = true_positives / (true_positives + false_negatives)

    return precision

#implementasi dari sistem
tab1, tab2 = st.tabs(["Implementasi","Modeling"])

with tab1:
    st.title("Spelling Correction")
    col1, col2 = st.columns([3, 1])  # Adjust the column ratios as needed
    user_input = col1.text_input("Masukkan Query")
    col2.write("")
    col2.write("")
    periksa = col2.button("Periksa")
    if periksa:
        st.write(user_input)
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
            # # Creating DataFrames
            # df_input = pd.DataFrame(np.array(tokenize), columns=["Input"])
            # df_result = pd.DataFrame(np.array(result.split()), columns=["Result"])

            # # Concatenating along columns (axis=1)
            # concatenated_df = pd.concat([df_input, df_result], axis=1)
            # hasil = np.array(concatenated_df)
            # hasil_itung = []
            # for index, row in concatenated_df.iterrows():
            #     dictionary = np.array(result.split())
            #     data_test = np.array(row).reshape(1, -1)
            #     precision, recall = calculate_precision_recall(data_test, dictionary)
            #     hasil_itung.append({
            #         'Precision': precision,
            #         'Recall': recall
            #     })

            # df_hasil = pd.DataFrame(hasil_itung)

            # # Calculate the average precision and recall
            # rata2recall = np.sum(df_hasil['Recall'].astype(float))  # Convert to float if not already
            # rata2precision = np.mean(df_hasil['Precision'].astype(float))  # Convert to float if not already

            # st.info(f"Nilai Recall: {rata2recall}") 
            # st.info(f"Nilai Precision: {rata2precision}") 

            mycursor = mydb.cursor()
    #Menampilkan judul dan konten
            
            mycursor.execute(f"SELECT * FROM data_berita_pariwisata__2_ WHERE judul LIKE '%{result}%' or konten LIKE '%{result}%'")
            judul_list = mycursor.fetchall()
            mycursor.close()
            if len(judul_list) == 0:
                st.warning("Tidak Ada Berita")
            else:
                for judul in judul_list:
                    with st.expander(f"{judul[1]} - {judul[2]}"):
                        # ... (your existing code)
                        st.markdown(f"<h3>{judul[1]}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<span>{judul[2]}</span>", unsafe_allow_html=True)
                        st.markdown(f"<p>{judul[4]}</p>", unsafe_allow_html=True)

with tab2:
    st.title('Modeling')
    # uploaded_file = st.file_uploader("Choose a file")
    # if uploaded_file:
    #     data = pd.read_csv(uploaded_file)
    # data = pd.read_csv('data_tes.csv')
    # dataset = pd.read_csv('data_query_pos - data_query_pos.csv')
    st.subheader("Dataset Pengujian query ")
    st.subheader("dataset pengujian query pos tagging")
    st.dataframe(data)
    st.dataframe(dataset)
    def bersihkan_teks(teks):
        # Menghapus huruf besar
        teks = teks.lower()

        # Menghapus karakter khusus dan tanda baca
        teks = re.sub(r'[^a-zA-Z0-9\s]', '', teks)

        # Menghapus tautan (link)
        teks = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', teks)

        return teks
        #untuk melakukan komputasi dari presisi dan recall dari correction 
    def komputasi(row):
        clean = bersihkan_teks(row[0])
        tokenize = clean.split()
        result = ""
        for input_word in tokenize:
            corrected_word = correct_spelling_deletion(input_word) 
            result += str(corrected_word) + " "   

        df_input = pd.DataFrame(np.array(tokenize), columns=["Input"])
        df_result = pd.DataFrame(np.array(result.split()), columns=["Result"])

        # Concatenating along columns (axis=1)
        concatenated_df = pd.concat([df_input, df_result], axis=1)
        hasil = np.array(concatenated_df)
        hasil_itung = []
        for index, row in concatenated_df.iterrows():
            dictionary = np.array(result.split())
            data_test = np.array(row).reshape(1, -1)
            precision = calculate_precision(data_test, dictionary)
            hasil_itung.append({
                'Precision': precision,
                # 'Recall': recall
            })

        df_hasil = pd.DataFrame(hasil_itung)

        # Calculate the average precision and recall
        #rata2recall = np.sum(df_hasil['Recall'].astype(float))  # Convert to float if not already
        rata2precision = np.mean(df_hasil['Precision'].astype(float))  # Convert to float if not already

        return result,rata2precision
    #untuk melakukan perhitungan koreksi kata presisi dan recall pada data yg uji  
    hasil_koreksi = []
    recall = []
    precision = []
    for index, row in data.iterrows():
        result,rata2precision = komputasi(row)
        hasil_koreksi.append(result)
        # recall.append(rata2recall)
        precision.append(rata2precision)

        for index, row in dataset.iterrows():
            result,rata2precision = komputasi(row)
            hasil_koreksi.append(result)
            # recall.append(rata2recall)
            precision.append(rata2precision)



    # menampilkan hasil koreksi kata pada data uji
    st.subheader("Hasil Koreksi kata pada Data test")
    hasil_koreksi_kata = pd.DataFrame(hasil_koreksi, columns = ['Hasil Koreksi Kata'])
    hasil_perbandingan = pd.concat([data,hasil_koreksi_kata],axis=1)
    st.dataframe(hasil_perbandingan)

    # menampilkan hasil koreksi kata pada data uji
    st.subheader("Hasil Koreksi kata pada Data test pos tagging")
    hasil_koreksi_kata = pd.DataFrame(hasil_koreksi, columns = ['Hasil Koreksi Kata'])
    hasil_perbandingan = pd.concat([dataset,hasil_koreksi_kata],axis=1)
    st.dataframe(hasil_perbandingan)

    # Hasil recall dan precision 
    st.subheader("Hasil Precision")
    # hasil_recall = pd.DataFrame(recall)
    # rata2recall = np.mean(hasil_recall[0])
    # st.info(f"Recall: {rata2recall}") 

    hasil_precision = pd.DataFrame(precision)
    rata2precision = np.mean(hasil_precision[0])
    st.info(f"Precision: {rata2precision}") 
    
#     else:
# st.warning('Masukkan Data terlebih dahulu')
                
                