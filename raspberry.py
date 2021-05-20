# ## 라즈베리 파이가 시작되면 바로 한 번 돌아가는 부분
# ## 이 이후에 파일이 꺼지지 않고 계속 돌고 있어야 실시간으로 작동 가능
# ## 어차피 이거말고 딱히 뭐 없을테니까... 라는 희망사항?

import os
import cv2
from keras import preprocessing
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import numpy as np

## 함수지정
def getImagepath(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    return imagePaths

def image_processing(image):
    img = preprocessing.image.load_img(image, color_mode = "grayscale")
    img = preprocessing.image.img_to_array(img).astype('float32') / 255
    img = cv2.resize(img, (200, 200))  # 사진 봐서 바꿔야함
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.expand_dims(img, axis=0)
    return img

def minValue(minvalue, new) :
    if new < minvalue :
        minvalue = new
    return minvalue

def Result(pred_list):
    pred_mean = sum(pred_list)/len(pred_list)
    pred_min = min(pred_list)
    pred_min_img_index = pred_list.index(pred_min)  # 유사도 가장 높은 이미지 선택
    result = [pred_mean, pred_min, user_dict[pred_min_img_index]]
    return result, pred_mean

def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(Y_true, D):
    Y_true= tf.cast (Y_true, tf.float32)
    margin = 1
    return K.mean(Y_true * K.square(D) + (1 - Y_true) * K.maximum((margin-D),0))

## 모델 로드
# ## 경로 라즈베리파이 환경에 맞게바꿔야함
model = load_model('/Users/spring/PycharmProjects/SmartCCTV/model_test/korean400_2.5.0_0.841.h5',
                   custom_objects={'contrastive_loss': contrastive_loss,
                                   'euclidean_distance': euclidean_distance})

## 등록된 user 이미지 전처리
    ## 경로 라즈베리파이 환경에 맞게바꿔야함
user_face_path = '/Users/spring/Desktop/user'
user_img_paths = getImagepath(user_face_path) # getImagepath : 이미지 경로 추출 함수

user_img = []
user_dict = {}

for idx, path in enumerate(user_img_paths):
    img = image_processing(path)

    user_dict[idx] = path
    user_img.append(img)

## Unknown 이미지 전처리
    ## 경로 라즈베리파이 환경에 맞게바꿔야함
unknown_face_path = '/Users/spring/Desktop/unknown'
unknown_img_paths = getImagepath(unknown_face_path)

unknown_img = []
unknown_dict = {}

if len(unknown_img_paths) == 0 :
    pass

else:
    for idx, path in enumerate(unknown_img_paths):
        img = image_processing(path)

        unknown_dict[idx] = path
        unknown_img.append(img)







## 카메라 켜지는 걸 인식하면 돌아가는 코드
# 얼굴 검출이 되는 15개 이미지 path 에서 가져오기 (== 얼굴 크롭) && 경로 수정 필요

path = '/Users/spring/Desktop/FD'

imagePaths = getImagepath(path)
# del imagePaths[2]

crop_dict = {}  # output1 ) 딕셔너리 {key - index : value - 원본 이미지 경로}
crop_img = []   # output2 ) 리스트 : 이미지 배열

for idx, path in enumerate(imagePaths):
    crop_face = image_processing(path)  # 이미지 전처
    crop_dict[idx] = path
    crop_img.append(crop_face)

## 역치 설정
## 저장리스트, 딕셔너리 초기화
threshold = 0.01
minvalue = 1
user_num = len(user_img) # 등록된 사용자 수
result = []
mean_result = {}

for num in range(user_num):  # 등록된 사용자 폴더에서 1장씩 가져옴
    pred_result = []
    for idx, img in enumerate(crop_img):  # Iot 이미지 1장씩 가져오기
        # 사용자와 Iot 이미지 비교 결과 저장 리스트
        pred_result.append(model.predict([img, user_img[num]])[0][0])

    pred, mean_value = Result(pred_result) # 결과 리스트 [평균, 최소값, 이미지경로]균, 평균값
    result.append(pred)   # 리스트에 pred 추가 ==> [[평균, 최소값, 이미지경로]]
    mean_result[mean_value] = num # 평균값 저장 딕셔너리 ==> {평균값 : 사용자 id}
    # 평균 중 최소 저장
    minvalue = minValue(minvalue, result[num][0])  # result[num][0] : 평균
# 등록된 사용자와 비교 후 가장 낮은 평균값으로 역치와 비교
# 역치보다 낮으면 특정인으로 분류
if minvalue <= threshold:
    # minvalue를 갖는 인덱스 찾아 특정인으로 분류
    user_id = mean_result[minvalue]
    save_image = cv2.imread(result[user_id][2])  # 이미지 경로 불러와서 배열로 변환 후 저장
    cv2.imwrite('record/User' + str(user_id) + '.jpg', save_image)
else:
    # unknown이랑 비교
    result = []
    mean_result = {}

    if len(unknown_img) > 0:
        for num in range(len(unknown_img)):
            pred_result = []
            for img in crop_img:
                pred_result.append(model.predict([img, unknown_img[num]])[0][0])
            pred, mean_value = Result(pred_result)
            result.append(pred)
            mean_result[mean_value] = num
            minvalue = minValue(minvalue, result[num][0])  # result[num][0] : 평균

        if minvalue <= threshold:
            unknown_id = mean_result[minvalue]  # unknown_id 보내는 방법
            save_image = cv2.imread(result[unknown_id][2])  # 이미지 경로 불러와서 배열로 변환 후 저장
            cv2.imwrite('record/Unknown' + str(unknown_id) + '.jpg', save_image)

        else:
            save_image = cv2.imread(crop_dict[0])
            cv2.imwrite(
                '/Users/spring/Desktop/unknown/' + str(len(unknown_img) + 1) + '_unknown.jpg',
                save_image)
            cv2.imwrite('record/Unknown.jpg', save_image)

    elif len(unknown_img) == 0:
        save_image = cv2.imread(crop_dict[0])
        cv2.imwrite('/Users/spring/Desktop/unknown/' + str(1) + '_unknown.jpg', save_image)
        cv2.imwrite('record/1_Unknown.jpg', save_image)