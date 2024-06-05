if option == '💯Testing  Tanpa POS Tagging' or option == '':
        #st.write("""# Proses Testing Data Query yang Berjumlah 100 Data""") #menampilkan halaman utama      
        
            #Fungsi untuk menghubungkan data yang ada di database 
        def correct_spelling_deletion(input_word):#koreksi kata berdasarkan kamus
        
            mycursor = mydb.cursor()#objek kursor pada koneksi database

            mycursor.execute("SELECT Unique_Words FROM kamus_berita_pariwisata_2")##untuk mendapatkan kamus didalam database

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

                #untuk melakukan komputasi dari presisi dari correction 
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
                hasil_koreksi_kata.to_excel("/xampp/htdocs/RISET_DIA/hasil_koreksi_tanpaPOS.xlsx")
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