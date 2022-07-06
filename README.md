## Requirements setup

Aşağıdaki gerekliliklerin kurulumundan önce bilgisayarda
opencv gerekliliklerinin kurulu olması gerekmektedir. Opencv ve gerekli
paketlerin kurulumları için döküman eklenecektir.

### 

### Sanal ortam oluşturulması

1. `conda create --name hancer_tracking`
2. `conda activate hancer_tracking`
3. `conda install python=3.8`
4. `conda install --file requirements.txt`

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

## Test

Resim test etmek için model dosyasını ve images klasörünü günceledikten sonra aşağıdaki komut ile test edilebilir.

`python test_image.py -i bi_72.jpg -w yolov4-tiny.weights -c yolov4-tiny.cfg -n obj.names -cn 0.1 -st 0.2 -io 0.1`
