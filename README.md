# Laporan Proyek Machine Learning - Prediksi Harga Rumah di Tehran, Iran - Ryan Badai Alamsyah

## Domain Proyek

### Latar Belakang

Tempat tinggal atau rumah merupakan salah satu dari sekian banyak kebutuhan primer bagi manusia. Oleh karena itu sangat penting untuk membuat sebuah perencanaan agar setiap keluarga dapat memiliki hunian pribadi. Perencanaan ini membutuhkan sebuah prediksi atau ramalan harga di masa yang akan datang. Namun, menentukan harga yang tepat untuk suatu properti bisa menjadi tugas yang kompleks dan membingungkan. Banyak faktor yang memengaruhi harga rumah, termasuk lokasi, ukuran, kondisi, fasilitas terdekat, dan tren pasar. Maka dari itu, saya membuat model prediksi harga rumah dengan menggunakan beberapa metode, yaitu KNN, Random Forrest, dan Boosting dengan tujuan memberikan panduan yang lebih baik kepada penjual dan pembeli dalam menentukan harga yang adil dan kompetitif untuk properti mereka.

## Business Understanding

### Problem Statements

- Faktor-faktor apa saja yang mempengaruhi harga rumah?
- Bagaimana perbandingan algoritma KNN, Random Forest, dan Boosting dalam memprediksi harga rumah?
- Berapa harga rumah berdasarkan hasil prediksi?

### Goals

- Mengetahui faktor apa saja yang mempengaruhi harga rumah.
- Mengetahui algoritma yang cocok digunakan dalam memprediksi harga rumah.
- Membuat model untuk mengetahui harga rumah berdasarkan dataset.

### Solution statements

- Menganalisis data agar dapat memahami korelasi antar variabel.
- Membandingakan tiga algoritma Machine Learning, yaitu KNN, Random Forest, dan Boosting.
- Menggunakan Akurasi dan MSE untuk mengetahui algoritma mana yang cocok digunakan dalam memprediksi harga rumah.

## Data Understanding
Dataset ini merupakan dataset yang berisi harga-harga rumah di Tehran, Iran dengan beberapa variabel yang mempengaruhi harga rumah tersebut. Dataset tersebut dapat diunduh di [Kaggle Repository](https://www.kaggle.com/datasets/mokar2001/house-price-tehran-iran/data)

Untuk informasi dari dataset tersebut, yaitu:
- Dataset berformat CSV
- Dataset mempunyai 8 fitur
- Dataset memiliki 3479 sample

#### Variabel-variabel pada Dataset House Price (Tehran, Iran) adalah sebagai berikut:
- Area: Berapa luas rumahnya
- Room: Jumlah kamarnya 
- Parking: Apakah terdapat tempat parkir
- Warehouse: Apakah terdapat gudangnya
- Elevator: Apakah terdapat elevator
- Address: Alamat rumahnya
- Price: Harga rumahnya
- Price(USD): Harga rumahnya dalam satuan dollar (USD)

#### Multivariate Analysis
![download (1)](https://github.com/RyanBadai/testing/assets/98163422/ff7d2ce9-431b-448c-a742-1e652ff854ed)

Dari matriks di atas menunjukkan bahwa fitur Room berkorelasi dengan Price dan fitur Werehouse dengan Parking juga berkorelasi. Yang dapat disimpulkan bahwa fitur Room, Wewhouse, dan Parking mempengaruhi harga rumah.

#### Penanganan Outliers
Dalam penelitian ini, metode IQR digunakan untuk mengatasi nilai-nilai yang tergolong sebagai outliers. IQR, atau Interquartile Range, merujuk pada rentang antara kuartil. Untuk memahaminya lebih baik, kita perlu mengingat konsep kuartil. Kuartil adalah tiga titik yang membagi distribusi data menjadi empat bagian yang seimbang. Separuh data berada di bawah kuartil pertama (Q1), satu perempat berada di bawah kuartil kedua (Q2), dan tiga perempat berada di bawah kuartil ketiga (Q3). Oleh karena itu, IQR dapat dihitung sebagai selisih antara Q3 dan Q1, yaitu IQR = Q3 - Q1.

Dalam proyek ini, metode IQR digunakan untuk mengidentifikasi dan mengatasi nilai-nilai outliers. Pertama-tama, outliers didefinisikan sebagai nilai-nilai di bawah Q1 (batas bawah, yang setara dengan 0,25) atau di atas Q3 (batas atas, yang setara dengan 0,75). Hasil dari Q3 dikurangi dengan hasil Q1, dan hasil ini disimpan dalam variabel baru yang dinamakan "insurance," yang merepresentasikan batas bawah hasil dari pengurangan Q1 dengan 1,5 kali nilai IQR. Sementara itu, batas atas dihitung dengan menambahkan 1,5 kali nilai IQR ke Q3.

## Data Preparation
- One Hot Encoding: Teknik untuk mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai 0 atau 1. Penulis menggunakan teknik One Hot Encoding, dikarenakan terdapat kolom yang bertipe kategorikal, yaitu True dan False. Kolom-kolom yang akan berubah adalah Parking, Warehouse, dan Elevator.
- Train Test Split: Proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 1406 dibagi menjadi 1335 data training dan 71 data testing.
- Normalization / Standarisasi: Teknik untuk membuat data menjadi seragam yang mempunyai skala relatif sama yang berguna untuk meningkatkan performa model.

## Model Development

Untuk algoritma yang digunakan dalam penelitian ini adalah sebagai berikut:

- KNN
 Algoritma pembelajaran mesin yang digunakan untuk masalah klasifikasi dan regresi. Konsep dasarnya adalah bahwa objek yang serupa cenderung berkumpul bersama. Algoritma ini bekerja dengan cara mengukur jarak antara titik data yang akan diprediksi dengan titik data lainnya dalam dataset. Kemudian, ia mengambil kategori mayoritas dari K titik terdekat (tetangga) untuk memprediksi kategori titik data yang akan diprediksi. Setelah dicoba dengan beberapa parameter dan nilainya, untuk parameter yang cocok digunakan dalam Model KNN ini adalah _n_neighbors = 1_

- Random Forest
Algoritma ensemble learning yang menggabungkan sejumlah besar pohon keputusan untuk meningkatkan akurasi prediksi. Setiap pohon dalam hutan membuat prediksi, dan hasil dari seluruh hutan diambil dengan cara mayoritas. Salah satu keunggulan Random Forest adalah kemampuannya untuk mengatasi overfitting dan memproses dataset dengan banyak fitur tanpa harus melakukan pemilihan fitur. Algoritma ini juga dapat memberikan pentingnya setiap fitur dalam prediksi. Setelah dicoba dengan beberapa parameter dan nilainya, untuk parameter yang cocok digunakan dalam Model Random Forest ini adalah _n_estimators = 50, max_depth = 16, dan random_state = 11_.

- Boosting
Boosting adalah teknik ensemble learning lainnya, tetapi berbeda dari Random Forest. Ada beberapa algoritma Boosting, termasuk AdaBoost, Gradient Boosting, dan XGBoost, yang semuanya bertujuan meningkatkan kinerja model. Prinsip dasar Boosting adalah menggabungkan sejumlah model lemah (model dengan akurasi sedang) menjadi model kuat. Prosesnya dilakukan dengan memberikan bobot lebih kepada titik data yang salah diprediksi oleh model sebelumnya, sehingga iterasi berikutnya akan fokus pada titik-titik data yang lebih sulit untuk diprediksi. Hasil dari seluruh iterasi diambil dengan bobot tertentu untuk menghasilkan prediksi akhir. Setelah dicoba dengan beberapa parameter dan nilainya, untuk parameter yang cocok digunakan dalam Model Boosting ini adalah _n_estimators = 25, learning_rate = 0.001, dan random_state = 11_

Penulis memilih untuk menggunakan tiga algoritma tersebut, yaitu K-Nearest Neighbors (KNN), Random Forest, dan Boosting. Keputusan ini didasarkan pada berbagai alasan yang kuat. Pertama, ketiga algoritma ini memiliki keunggulan masing-masing, seperti kecepatan dan kemampuan menangani dataset besar. Kedua, penggunaan tiga algoritma berbeda membantu mengatasi bias dan memberikan hasil yang lebih konsisten. Ketiga, evaluasi kinerja algoritma ini akan memberikan wawasan tentang model mana yang paling cocok untuk tugas ini. Keempat, KNN memberikan pemahaman awal yang mudah diinterpretasi, sementara Random Forest dan Boosting dapat memberikan pemahaman yang lebih dalam tentang faktor-faktor yang memengaruhi harga rumah. Terakhir, dengan tiga algoritma ini, penulis memiliki fleksibilitas dalam menangani variasi dalam dataset dan tantangan yang mungkin muncul selama penelitian. Dengan demikian, penggunaan KNN, Random Forest, dan Boosting diharapkan dapat menghasilkan prediksi harga rumah yang lebih akurat dan dapat diandalkan.

## Evaluation

Dalam penelitian ini, penulis menggunakan metrik akurasi dan MSE untuk mengevaluasi model. Dimana akurasi adalah metrik yang digunakan untuk mengukur sejauh mana model machine learning atau klasifikasi berhasil memprediksi kelas atau label data yang benar. Ini adalah salah satu metrik yang paling umum digunakan dalam evaluasi model klasifikasi. Sedangkan MSE adalah suatu metrik yang digunakan untuk mengukur sejauh mana hasil prediksi dari suatu model regresi atau machine learning mendekati nilai yang sebenarnya. MSE mengukur rata-rata dari kuadrat selisih antara nilai sebenarnya (y) dan prediksi (ŷ) dari model. Untuk hasil dari metriks akurasi dan MSE adalah sebagai berikut:

- Akurasi

| Model | Akurasi | 
| :---------: | :---------: |
| KNN | 0.74869 | 
| Random Forest | 0.590423 |
| Boosting | 0.260317 | 

- Visualisasi MSE
![download2](https://github.com/RyanBadai/testing/assets/98163422/d93f4f79-34f4-41b4-b6ed-fc3ecb63bd5b)

- Hasil

| | y_true | prediksi_KNN | prediksi_Random Forest | prediksi_Boosting |
| :---------: | :---------: | :---------: | :---------: | :---------: |
| **2363** | 1.100000e+09 | 8.960000e+08 | 2.699430e+09	| 3.115507e+09 |
| **906** | 4.200000e+09 | 4.600000e+09 | 5.089435e+09 | 5.485642e+09 |
| **39** | 4.830000e+09 | 3.950000e+09 | 4.755876e+09 | 5.485642e+09 |
|**3440** | 4.000000e+09 | 3.600000e+09 | 3.719135e+09 | 5.408575e+09 |
| **803** | 3.050000e+09 | 4.000000e+09 | 2.853497e+09 | 3.115507e+09 |

Hasil evaluasi model menggunakan metrik akurasi dan MSE menunjukkan bahwa model KNN memiliki akurasi yang tinggi dan nilai MSE yang rendah dalam memprediksi harga rumah. Kesimpulannya, algoritma KNN adalah pilihan terbaik untuk studi kasus ini.

### Referensi
[1] [Saiful, A. (2021) ‘House Price Prediction Using Web Scrapping and Machine Learning With Linear Regression Algorithm’, JATISI (Jurnal Teknik Informatika dan Sistem Informasi), 8(1), pp. 41–50. doi:10.35957/jatisi.v8i1.701.](https://jurnal.mdp.ac.id/index.php/jatisi/article/view/701)

[2] [Assudani, P. and Wankhede, C. (2022) ‘Analysing the factors influencing the house prices and studying house price prediction methods’, International Journal of Next-Generation Computing [Preprint]. doi:10.47164/ijngc.v13i5.952. ](https://ijngc.perpetualinnovation.net/index.php/ijngc/article/view/952)

