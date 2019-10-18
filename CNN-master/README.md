# Evrişimsel Sinir Ağları (Convolutional Neural Network) Uygulaması

Bu projede verilen 10 sınıflı bir görüntü veri kümesinin CNN yöntemi ile
sınıflandırılmasını sağlayan bir sistem tasarlanmıştır. Proje boyunca gerçekleştirilen adımlar ve
kullanılan fonksiyonlar sırasıyla anlatılmıştır.

<b>getImageNamesAndClassLabels:</b> Projenin ilk adımıdır. Dataset klasöründeki tüm resimler
okunmaktadır ve her resim image_name listesine, her resmin sınıfı label listesine eklenmektedir
ve fonksiyondan bu iki liste döndürülmektedir.

<b>splitDatasetTrainAndTest:</b> Bu fonksiyona input olarak getImageNamesAndClassLabels
fonksiyonundan dönen resim dizinleri ve resim sınıfları verilmektedir. Scikit-learn
kütüphanesinin bir fonksiyonu olan train_test_split fonksiyonu ile tüm resimlerin %20’si test
resmi olarak ayrılmıştır. Bu fonksiyondan train_images, test_images, train_labels, test_labels
listeleri döndürülmektedir.

<b>findUniqueLabels:</b> Bu fonksiyon Dataset klasörü içinde bulunan sınıfları liste olarak
döndürmektedir.

<b>createFolders:</b> Bu fonksiyon eğitim ve test işlemleri için veri setini belli bir düzene göre
düzenlemektedir

<b>createImageForTrainingAndTesting:</b> Bu fonksiyon ile eğitimde kullanılacak veri seti
düzenlenmektedir. Bu fonksiyon sonucunda train_data ve test_data olmak üzere iki adet liste
oluşmaktadır. Listeler Data klasöründeki düzene göre oluşturulmaktır. Listenin her değeri,
resim dizisi ve resmin sınıf indisinden oluşmaktadır.

<b>divideDataAndLabel:</b> Bu fonksiyonda createImageForTrainingAndTesting fonksiyonundan
dönen listeler veri ve veri sınıfı olmak üzere iki listeye ayırılmaktadır.
plot_confusion_matrix: Scikit-Learn kütüphanesi için hazırlanmış dokümantasyondan
alınmıştır. Confusion Matrix çizimi için kullanılmıştır.

<b>predictImages:</b> Dizini verilen bir resim için sınıf tahmini yapılması sağlayan fonksiyondur.
Tahminlenen sınıfın ismini döndürmektedir.

<b>putText:</b> Dizini verilen resim için predictImages fonksiyonundan dönen sınıf ismini resmin
üzerine yazılması için kullanılmıştır. Bu resim Output klasöründe saklanmaktadır.

<b>draw_confusion_matrix:</b> Confusion Matrix çizimi için gerekli ön işlemler yapılmaktadır.

<b>testCNNModel:</b> Test klasöründe bulunan resimler için sınıf tahmini yapılmaktadır. Loss ve
accuary değerleri yazdırılmaktadır.

<b>trainCNNModel:</b> Bu fonksiyon içinde eğitim işlemi gerçekleşmektedir. Her epoch sonrası
oluşan model checkpoints klasörüne kaydedilmektedir.

<b>ESA Modeli:</b> CNN mimarisi aşağıdaki resimde gösterildiği gibidir. Sırasıyla arka arkaya
evrişim katmanı, aktivasyon katmanı ve havuzlama katmanı 5 kere tekrar edilmiştir. Sonrasında
3 adet tam-bağlı katman kullanılmış ve en son da softmax sınıflandırma katmanı kullanılmıştır.
Ezberden kaçınmak için dropout kullanılmıştır. Dropout sırasıyla m5, FC1 ve FC2
katmanlarından sonra kullanılmıştır. Aktivasyon katmanın amacı evrişim katmanı uygulanırken
elde edilen verilerin doğrusal ortamdan doğrusal olmayan ortama aktarılıp analiz edilebilmesini
sağlamaktır. Havuzlama katmanının amacı elde edilen içeriğin indirgenmesidir. Tam bağlı
katman ise kendisinden önce yer alan bütün nöronlardaki değerlerin kendisinden sonra
oluşturulan tüm nöronlarla eşleşmesini sağlayıp çıkarılabilecek en fazla sayıda ayırt edici
özelliklerin elde edilmesini sağlamaktadır.

<img src="https://github.com/seymenmurat16/CNN/blob/master/1.PNG"/>


### Örnek
<img src="https://github.com/seymenmurat16/CNN/blob/master/2.PNG"/>

### Confusion Matris
<img src="https://github.com/seymenmurat16/CNN/blob/master/3.PNG"/>
