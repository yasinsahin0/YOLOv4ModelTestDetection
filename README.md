## Requirements setup

Aşağıdaki gerekliliklerin kurulumundan önce bilgisayarda
opencv gerekliliklerinin kurulu olması gerekmektedir. Opencv ve gerekli
paketlerin kurulumları için döküman eklenecektir.

### 

### Sanal ortam oluşturulması

1. `conda create --name yolo_test`
2. `conda activate yolo_test`
3. `conda install python=3.8`
4. `conda install --file requirements.txt`

NOT : Permission hatası için arch condanın bulunduğu klasöre `sudo chmod 777 archiconda3` diyerek tam izin verilmelidir.

Gerekliliklerin yüklenmesinde bir hata alınırsa yeniden
sanal ortam oluşturarak aşağıdaki kurulum adımlarını sırasıyla
ilerletebilirsiniz.

### 

### Opencv kurulumu

1. `conda install -c conda-forge opencv`
2. `conda install -c conda-forge/label/gcc7 opencv`
3. `conda install -c conda-forge/label/broken opencv`
4. `conda install -c conda-forge/label/cf201901 opencv`
5. `conda install -c conda-forge/label/cf202003 opencv`

## Jetson

Jetson bağlı usb üzerinden dosyaları aktarmak için aşağıdaki adımlar takip edilebilir.

1. root # cd media/$USER/usbdiskname
2. root # cd ../../
3. `root # sudo cp -r media/$USER/usbdiskname/tasinacakdosya home/$USER/Desktop`

### Jetson anaconda kurulumu

1. `wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh`
2. `sudo sh Archiconda3-0.2.3-Linux-aarch64.sh`

## Test

Resim test etmek için model dosyasını ve images klasörünü günceledikten sonra aşağıdaki komut ile test edilebilir.

`python test_image.py -i bi_72.jpg -w yolov4-tiny.weights -c yolov4-tiny.cfg -n obj.names -cn 0.1 -st 0.2 -io 0.1`
