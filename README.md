# ** yolov5를 이용한 object detection 개인 실습 기록 **

## 1. 환경 설정.

 ### device 
    운영체제 : window 10
    CPU : i5-7500
    RAM : 16Gb
    GPU : GeForce GTX 1060 6G

    가상환경에서 install 진행.
    python : 3.8.12
    cuda : 11.1
    cuDNN : 0.8.5 
    opencv : 4.5.5 
    pytroch : 1.8.0   --> pytroch-gpu install

    (버전 확인)
    spyder 사용.
    import cv2
    import torch

    print(torch.__version__)  
    print(cv2.__version__)

    # 파이토치 사용가능 gpu 확인 #
    torch.cuda.is_available()               ----> True
    torch.cuda.get_device_name(0)      ----> 'GeForce GTX 1060 6GB'
    torch.cuda.device_count()	        ----> 1


## 2. YOLOV5 clone 진행.

    사용하고자하는 경로에 git clone https://github.com/ultralytics/yolov5.git

    (오류가 발생하면 conda install git)

    anaconda prompt 관리자 모드 -> 가상환경 activate -> yolov5 경로로 이동.

    yolov5경로로 이동하게 되면 requirements에 yolov5를 사용하기 위한 여러 인스톨 파일들이 있음.

    anaconda prompt yolov5경로에 

    -> pip3 install -r requirements.txt

    yolov5 사용 준비 완료.

    진행 과정에서 install 문제로 오류가 발생한다면 3번부터 진행.



## 3. DATA SET 준비

     google에 labelimg 검색

     보기 쉽게 폴더관리를 하기위해 yolov5 폴더 이전에 상위 폴더를 만들었고, yolov5와 같은 경로에 labelimg clone 진행함.

     anaconda prompt 에서 가상환경 이동 
     사용하고자 하는 경로에 git clone https://github.com/tzutalin/labelimg.git
     conda install pyqt =5
     conda install -c anaconda lxml
     pyrcc5 -o libs/resources.py resources.qrc

     설치한 labelimg 폴더에서 
     python labelImg.py -> 그림에대한 라벨링 진행.

     본인은 유튜브에서 자동차 드라이브 영상, 도로주행 영상, 드라이브 영상 등을 다운받아서 라벨링을 진행 하였으나,
     직접 라벨링하는 데이터수는 오픈소스 데이터수를 커버하기 힘들었고, train의 결과 또한 정확도가 높지않을것으로 판단되어

     roboflow에서 제공하는 opensource dataset을 사용하였음.

     Ref : https://public.roboflow.com/object-detection
     Ref : https://www.youtube.com/watch?v=y3FkRXZqE2s 에 자세하게 설명이 나와있음.


## 4. yolov5 사용을 위한 yaml 파일 셋팅, 경로 셋팅.

     차량 주행을 위한 comtom data 를 training 시키기 위해 dataset images와 images에 대한 labels를 완성했다면

     yaml 파일을 생성해주어야함.

     share -> yolov5/data -> forme.yaml

     forme 파일은 필자가 만든 yaml 파일이며, dataset의 path와 train, validation을 위한 파일 경로를 지정해주어야하고
     detect할 feature 수와 이름을 정의해 주어야함.

     roboflow에서 제공하고 있는 opensouce data는 총 11개의 detection feature가 설정되어 있음.

     train을 위해 dataset을 train data 와 val data로 나눠주어야함.
     (overfitting방지, input에대한 정확도 향상을 위함) 

     data_split.txt 는 data_split.py작성을 위해 test했던 내용들이며
     최종적으로 data_split.py를 사용하여 yolov5를 설치했던 dataset 경로에 split한 data의 list를 txt파일로 생성해주었음.

     여기서 생성된 파일은 yolov5 training시 yaml파일에 의해 txt파일이 읽히게 되고, train, val에 사용할 이미지파일을 지정해줌.


## 5. model training 

     anaconda prompt -> 가상환경 activate -> yolov5 설치 경로 이동

     yolov5 폴더 내부의 train.py파일이 사용됌.

     python train.py --data ./data경로입력/forme.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --batch 8 --worker 2 --epochs 100 --name final

     설정할 수 있는 파라미터를 확인하기 위해 train.py 내부의 파라미터를 확인하였고 GPU사양에 맞는 batch와 worker를 설정하였음.
     training은 

     CPU : i9 10900
     GPU : GeForce 3060 12G
     RAM : 64G
     사양의 컴퓨터에서 실행되었음. 
     실행시 GPU RAM의 11G정도 사용했고 총 29800개의 이미지 데이터를 train -> 20860,  val -> 8940개로 나누어 진행되었으며
     80시간정도 소요됌.

     트레이닝이 완료되면 yolov5 -> runs -> triain -> final(설정이름) -> weights -> best.pt 생성.

     best.pt파일을 사용하여  detect 실행.

## 6. model detect

     생성한 detect파일로 detection 실행.

     python detect.py --source ../dataset/video3/ --weights ./runs/train/final10/weights/best.pt

     source는 데이터 파일의 경로
     weights는 생성한 pt파일 경로.

     source에 1 을 입력하면 webcam 연동가능.
     실제로 웹캠에 연동하여 아이패드로 자동차 사진을 보여줬을때 detect되는걸 확인할 수 있음.


참고 영상

Ref : https://public.roboflow.com/object-detection

Ref : https://www.youtube.com/watch?v=y3FkRXZqE2s 에 자세하게 설명이 나와있음.


## 7. Result 

<img width="80%" src = "https://user-images.githubusercontent.com/97965904/150042936-f4bc6697-a702-43a9-a3ca-36dd5a53b027.gif"/>






