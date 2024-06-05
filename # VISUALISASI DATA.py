#   VISUALISASI DATA
def visualisasi(data):
    # Membaca file CSV
    df = pd.read_csv('/xampp/htdocs/RISET_DIA/Data_Berita_Pariwisata_skripsi.csv')

    all_text_positive = ' '.join(str(word) for word in df['Konten'])
    wordcloud = WordCloud(max_font_size=260, max_words=50, width=1000, height=1000, mode='RGBA', background_color='black').generate(all_text_positive)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0) 
    st.pyplot(plt)

if selected == 'Visualisasi':
    st.write("""## Visualisasi""") #menampilkan judul halaman dataframe
    def visualisasi(data):
    else st.button('Process'):
    # Membaca file CSV
        df = pd.read_csv('/xampp/htdocs/RISET_DIA/Data_Berita_Pariwisata_skripsi.csv')

    all_text_positive = ' '.join(str(word) for word in df['Konten'])
    wordcloud = WordCloud(max_font_size=260, max_words=50, width=1000, height=1000, mode='RGBA', background_color='black').generate(all_text_positive)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0) 
    st.pyplot(plt)
    uploaded_file = st.file_uploader("Please choose a CSV file", type='csv')
    if uploaded_file is not None:
        if st.button('Process'):
            df = pd.read_csv(uploaded_file, encoding='unicode_escape', sep=';')
            option = st.selectbox('Silahkan pilih:', ('Artikel Pariwisata POS Tagging', 'Artikel Pariwisata Tanpa POS Tagging'))
            if option == 'Artikel Pariwisata POS Tagging':
                df = df.loc[df['Label'] == 1]
                st.dataframe(df)
                visualisasi(uploaded_file)
            elif option == 'Artikel Pariwisata Tanpa POS Tagging':
                df = df.loc[df['Label'] == 1]
                st.dataframe(df)
                visualisasi(uploaded_file)
#implementasi dari sistem
#tab1, tab2 = st.tabs(["Implementasi","Modeling"])
if selected == 'Visualisasi':
    st.write("""## Visualisasi""") #menampilkan judul halaman dataframe
    uploaded_files = st.file_uploader("Please choose a CSV file",type = ['csv'])
    option = st.selectbox('Silahkan pilih:',('Artikel Pariwisata POS Tagging','Artikel Pariwisata Tanpa POS Tagging'))
    if selected == "Ulasan Positif":
        if uploaded_files is not None:
            if st.button('Process'):
                df = pd.read_csv(uploaded_files,encoding='unicode_escape',sep=';')
                df=df.loc[df['Label']==1]
                st.dataframe(df)
                visualisasi(df)
    def visualisasi(data):
    # Membaca file CSV
    data = pd.read_csv('/xampp/htdocs/RISET_DIA/Data_Berita_Pariwisata_skripsi.csv')

    all_text_positive = ' '.join(str(word) for word in data['Konten'])
    wordcloud = WordCloud(max_font_size=260, max_words=50, width=1000, height=1000, mode='RGBA', background_color='black').generate(all_text_positive)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0) 
    st.pyplot(plt)
    return data

# Memanggil fungsi visualisasi
data = visualisasi('/xampp/htdocs/RISET_DIA/Data_Berita_Pariwisata_skripsi.csv')