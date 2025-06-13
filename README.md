🎨 Görüntü Renk Analizi ve İyileştirme Aracı

💡 Proje Hakkında
Bu proje, Python ve OpenCV kütüphanelerini kullanarak görüntülerdeki baskın renkleri tespit etme, histogram eşitleme ile görüntü parlaklığını iyileştirme ve çeşitli renk uzaylarında (HSV, RGB, YCrCb, LAB, Luv) detaylı renk analizi yapma yeteneğine sahip gelişmiş bir görüntü işleme aracıdır. Kullanıcı dostu arayüzü sayesinde, bir klasördeki tüm görüntüler üzerinde bu işlemleri toplu olarak gerçekleştirebilirsiniz.

✨ Özellikler
Toplu İşleme: Belirtilen bir klasördeki tüm desteklenen görüntü dosyalarını (.png, .jpg, .jpeg, .gif, .bmp) otomatik olarak işler.
Histogram Eşitleme: Görüntülerin parlaklık dağılımını iyileştirerek daha net ve dengeli görseller elde etmenizi sağlar. Eşitlenmiş görüntüler ayrı bir Equized_Images alt klasörüne kaydedilir.
ColorThief Entegrasyonu: Eşitlenmiş görüntülerden baskın renkleri ve renk paletlerini çıkararak görsel analizi zenginleştirir.
HSV Tabanlı Renk Tespiti: Siyah, beyaz ve gri tonlarını HSV renk uzayında özel maskeleme teknikleriyle tespit eder ve görselleştirir.
Çoklu Renk Uzayı Analizi: RGB, HSV, YCrCb, LAB ve Luv renk uzaylarındaki histogramları grafikler halinde göstererek renk dağılımları hakkında derinlemesine bilgi sunar.
Baskın Renk Algılama (Özel Algoritma): Geliştirilmiş özel algoritmalarla görüntünün genel baskın rengini belirler ve görsel çıktılar sunar.
Kullanıcı Dostu Arayüz: Tkinter modülü sayesinde klasör ve dosya seçimi işlemleri kolay ve etkileşimlidir.
🛠️ Kullanılan Teknolojiler
Python: Projenin temel programlama dili.
OpenCV (cv2): Görüntü işleme görevleri için ana kütüphane.
NumPy: Sayısal işlemler ve dizi manipülasyonu için.
Matplotlib: Histogram ve diğer görselleştirmeler için.
ColorThief: Görüntülerden baskın renkleri ve paletleri çıkarmak için.
Pillow (PIL): ColorThief'in bir bağımlılığı olabilir, görüntü yükleme ve işleme desteği sağlar.
Tkinter: Kullanıcıdan klasör ve dosya yolu almak için temel GUI araç takımı.

🚀 Kurulum
Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:
Conda Ortamını Oluşturun ve Aktif Edin: Projeyi sorunsuz çalıştırmak için environment.yml dosyasını kullanarak bir Conda ortamı oluşturun: 
BASH: conda env create -f environment.yml 
Ortam aktif edildikten sonra uygulamayı başlatabilirsiniz: 
BASH: python ColorDetection.py

💡 Kullanım
Klasör Seçimi:
Program başladığında, işlemek istediğiniz görüntü dosyalarının bulunduğu klasörü seçmenizi isteyen bir pencere açılacaktır. İlgili klasörü seçin.

İşlem Başlangıcı:
Klasörü seçtikten sonra, program klasördeki tüm uyumlu görüntü dosyalarını sırayla işlemeye başlayacaktır. Her görüntü için aşağıdaki adımlar uygulanır:

Histogram Eşitleme: Görüntülerin parlaklığı eşitleyecek ve [Seçilen_Klasör]/Equized_Images/ altında Equized_[Orijinal_Ad].jpg formatında kaydedecektir.
ColorThief Analizi: Eşitlenmiş görüntü üzerinde baskın renk ve renk paleti tespiti yapılır, sonuçlar terminale yazdırılır.
HSV Maske Görselleştirme: Orijinal görüntü üzerinde siyah, beyaz ve gri maskelerin tespitini ve görselleştirmesini içeren bir dizi OpenCV penceresi açılacaktır. Bu pencereleri kapatmak için herhangi bir tuşa basın.
Baskın Renk Tespiti (Özel Algoritma): Görüntünün genel baskın rengi terminale yazdırılır.
