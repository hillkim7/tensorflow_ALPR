import sys, os
import cv2
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import img_utils

model_file_path = './my_models/number_digits'
src_path = './my_images/vehicle_number_plates'
dst_path = './my_results'

if len(sys.argv) < 2:
  sys.exit('Usage: %s <%s/image_file.png>' % (sys.argv[0], src_path))

if not os.path.isdir(dst_path): os.mkdir(dst_path)

image_file = sys.argv[1]

learning_rate = 0.0001
training_epochs = 100
batch_size = 100

# set image sizes
x_size = 25
y_size = 35

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, y_size * x_size])
X_img = tf.reshape(X, [-1, y_size , x_size, 1])   # img 28x28x1 (black/white) => 35x25x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 35, 25, 1)
W1 =  tf.get_variable('w1', shape=[3, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 35, 25, 32)
#    Pool     -> (?, 18, 13, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 35, 25, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 35, 25, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 18, 13, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 18, 13, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 18, 13, 32)
W2 =  tf.get_variable('w2', shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 18, 13, 64)
#    Pool      ->(?, 9, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 18, 13, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 18, 13, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 9, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 9, 7, 64), dtype=float32)
'''

# L3 ImgIn shape=(?, 9, 7, 64)
W3 =  tf.get_variable('w3', shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 9, 7, 128)
#    Pool      ->(?, 5, 4, 128)
#    Reshape   ->(?, 5 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 128 * 5 * 4])
'''
Tensor("Conv2D_2:0", shape=(?, 9, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 9, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 5, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 5, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2560), dtype=float32)
'''

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 5 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# b4 = tf.Variable(tf.random_normal([625]))
b4 = tf.Variable(tf.zeros([625]))

L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
# b5 = tf.Variable(tf.random_normal([10]))
b5 = tf.Variable(tf.zeros([10]))
logits = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

#------------------------------------------------------
def getIntersection(x,y,w,h,_x,_y,_w,_h):
    if (x > _x + _w): return 0;
    if (x + w < _x): return 0;
    if (y > _y + _h): return 0;
    if (y + h < _y): return 0;

    n_x = max(x, _x);
    n_y = max(y, _y);
    n_w = min(x + w, _x + _w) - n_x;
    n_h = min(y + h, _y + _h) - n_y;

    return n_w*n_h;

def make_background(filename,x_size,y_size):
    new_img = Image.new("RGB",(x_size,y_size),"white")
    im = Image.open(filename)
    #------------------------------------------------#
    # if im.size[0] / im.size[1] > 2.5 : #긴 번호판
    #     basewidth = x_size
    #     wpercent = (basewidth / float(im.size[0]))
    #     hsize = int((float(im.size[1]) * float(wpercent)))
    #     im = im.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    # else:
    #     baseheight = y_size
    #     hpercent = (baseheight / float(im.size[1]))
    #     wsize = int((float(im.size[0]) * float(hpercent)))
    #     im = im.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
    #------------------------------------------------#
    im.thumbnail((x_size,y_size),Image.ANTIALIAS)
    load_img = im.load()
    load_newimg = new_img.load()
    i_offset = (x_size-im.size[0])/2
    j_offset = (y_size-im.size[1])/2
    for i in range(0, im.size[0]) :
        for j in range(0, im.size[1]) :
            load_newimg[i+i_offset, j+j_offset] = load_img[i,j]

    new_img = np.array(new_img)
    new_img = new_img[:, :, ::-1].copy()
    # new_img.save(outfile, "JPEG")
    # new_img.show()
    return new_img


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def draw_lines(img, lines, color=[0, 0, 255], thickness=2): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     # print(x1, y1, x2, y2,' => ', abs(x1 - x2))
    #
    # return line_img, lines
    if lines  is None: lines=[]
    return lines

def weighted_img(img, initial_img, α=1, β=1., λ=0.): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def find_degree(image):
    # height, width = image.shape[:2]  # 이미지 높이, 너비
    height, width, _ = image.shape

    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백이미지로 변환
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)  # Blur 효과
    # canny_img = canny(blur_img, 70, 210)  # Canny edge 알고리즘
    canny_img = auto_canny(blur_img, sigma=0.33)
    # 화면 하단 3/4 영역만 대상으로 함
    vertices = np.array([[(0,height),(0, (height/4)*2), (width, (height/4)*2), (width,height)]], dtype=np.int32)
    ROI_img = region_of_interest(canny_img, vertices) # ROI 설정

    # hough_img = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환
    result_lines  = hough_lines(ROI_img, 1, 1 * np.pi/180, 40, 110, 17) # 허프 변환
    # result_lines  = hough_lines(ROI_img, 1, 1 * np.pi/180, 40, 90, 17) # 허프 변환
    len_result = len(result_lines)
    if len_result==0 : return 180
    result_lines = np.squeeze(result_lines)
    # print(result_lines, len(result_lines))

    if len_result>1:
        #정렬 y1, y2의 평균값 - 제일 큰(하단) 라인 만 선택
        max_y = 0
        temp_line=[]
        for a_line in result_lines:
            mean_y = (a_line[1]+a_line[3])/2
            if mean_y > max_y :
                max_y = mean_y
                temp_line = a_line
        result_lines=[]
        result_lines=temp_line
        # print(result_lines)
    # print('-------------------------')
    line_arr = np.squeeze(result_lines)
    line_arr = line_arr.reshape(-1,4)
    # print(line_arr)
    # print('-------------------------')
    # 기울기 구하기
    # print(line_arr.shape)
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
    # slope_degree = (np.arctan2(line_arr[1] - line_arr[3], line_arr[0] - line_arr[2]) * 180) / np.pi
    # print(slope_degree)

    # result = weighted_img(hough_img, image) # 원본 이미지에 검출된 선 overlap
    # cv2.imshow('result',result) # 결과 이미지 출력
    # cv2.waitKey(0)
    return slope_degree



def ExtractNumber(sess, filename):
    print(filename)
    s = os.path.splitext(filename)  #('test/1_0001', '.jpg')
    fname = os.path.split(s[0])  # 1_0001
    fext = os.path.split(s[1]) # .jpg
    if fext[1]=='.png': fext=('','.jpg')
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if debug >= 5:
    # 이미지 원본 보여주기
        cv2.imshow('1',img)
        cv2.waitKey(0)
    wh_h, wh_w, _ = img.shape

    # ------------------------
    # 기울기 계산
    d = find_degree(img)
    # print(d)
    d_gap = int(abs(180 - int(d)))
    # print(d_gap)
    # rotate 번호판이 조금 기울어져 있을 때 보정
    if d_gap > 0:
        if d < 180: dir=-d_gap
        else : dir = d_gap
        M1 = cv2.getRotationMatrix2D((wh_w/2, wh_h/2),dir,1)  #양수 : 반시계, 음수 : 시계방향
        img = cv2.warpAffine(img,M1,(wh_w, wh_h))
        if debug >= 5:
            cv2.imshow('1-1',img)
            cv2.waitKey(0)
    #--------------------------------
    # 신형 번호판 : 길이를 300으로 보정한 후 글자 폭, 높이, 비율 계산
    basewidth = 300  # x_size
    wpercent = (basewidth / float(wh_w))
    hsize = int((float(wh_h) * float(wpercent)))
    img = cv2.resize(img, (basewidth, hsize))
    wh_h, wh_w = img.shape[:2]
    wh_rect_area = wh_w * wh_h  # area size
    wh_aspect_ratio = float(wh_w) / wh_h
    # 번호판 크기 출력
    if debug >= 3: print(wh_rect_area, wh_aspect_ratio, wh_w, wh_h)

    copy_img = img.copy()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if debug >= 5:
        cv2.imshow('2', img2)
        cv2.waitKey(0)

    blur = cv2.GaussianBlur(img2, (3, 3), 0)

    if debug >= 5:
        cv2.imshow('3', blur)
        cv2.waitKey(0)

    canny = auto_canny(blur, sigma=0.33)

    if debug >= 4:
        cv2.imshow('4', canny)
        cv2.waitKey(0)

    cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #추출된 영역 정렬 - x축 기준
    box0 = []
    for i in range(len(contours)):  ##Buble Sort on python
        cnt = contours[i]
        box0.append(cv2.boundingRect(cnt))
    for i in range(len(box0)):  ##Buble Sort on python
        for j in range(len(box0) - (i + 1)):
            if box0[j][0] > box0[j + 1][0]:
                temp = box0[j]
                box0[j] = box0[j + 1]
                box0[j + 1] = temp
    box1 = []
    box2 = []
    f_count = 0
    select = 0
    plate_width = 0
    _x, _y, _w, _h = 0, 0, 0, 0
    # for i in range(len(contours)):
        # cnt = contours[i]
        # area = cv2.contourArea(cnt)
        # x, y, w, h = cv2.boundingRect(cnt)
    for i in range(len(box0)):
        x, y, w, h = box0[i]
        rect_area = w * h  # area
        aspect_ratio = float(w) / h  # ratio = width/height
        #    x축 90-125  y축 17-60 사이면 일단 패스
        cond2 = (x>=90 and (x+w) <=125 and y >= 17 and (y+h) <=60)
        # 넓이와 비율로 거른다
        cond1 =  not(w>40 or (rect_area<40 and (wh_h/5)*4 < y))  #True # ((aspect_ratio>=0.35)and(aspect_ratio<=0.9)and(rect_area>=500)and(rect_area<=1400))
        #가장자리 제외
        # cond3 = not(basewidth-30 < x+w)
        #가장자리(4변) 제외 x축 16, y축 6 => 20
        # cond3 = not(wh_w-16 < x+w or 16 > x or wh_h-6 < y+h or 6 > y )
        cond3 = not((wh_w-16 < x+w and w < 15) or (16 > x and (h<14 or w<10)) or (wh_h-6 < y+h and h<15) or (6 > y and w < 15) )

        #추출된 문자 가능 영역의 넓이,비율, 좌표 출력
        # if debug >= 3: print(i, rect_area, aspect_ratio, x, y, w, h,'[',x, y, x+w, y+h,']',cond1,cond2,cond3)
        if (cond2 or (cond1 and cond3)):
        # if 1:
            if (abs(x - _x) <= 2 and abs(y - _y) <= 2 and abs(w - _w) <= 2 and abs(
                    h - _h) <= 2): continue  # 중복제거 -> 개선 필요 - 겹치면 제거
            if debug >= 3: cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  #사각형 그리기 - 후보 단계 검증시 사용
            # box1.append(cv2.boundingRect(cnt))
            box1.append(box0[i])
            _x, _y, _w, _h = x, y, w, h
            #확정된 문자 영역의 넓이,비율, 좌표 출력
            if debug >= 3:
                print('--->', i, rect_area, aspect_ratio, x, y, w, h,'[',x, y, x+w, y+h,']')
                cv2.imshow('5',img)
                cv2.waitKey(0)

    #추출된 영역 정렬 - x축 기준
    for i in range(len(box1)):  ##Buble Sort on python
        for j in range(len(box1) - (i + 1)):
            if box1[j][0] > box1[j + 1][0]:
                temp = box1[j]
                box1[j] = box1[j + 1]
                box1[j + 1] = temp

    # 추출된 영역이  x축 기준을 겹치면 합치기
    _x, _y, _w, _h = 0, 0, 1, 1
    temp_cnt = 0
    find_num = len(box1)
    for i in range(find_num):
        x, y, w, h = box1[i][0], box1[i][1], box1[i][2], box1[i][3]
        if debug >= 2: print('all', x, y, w, h, '[', x, y, x + w, y + h, ']', w*h, w/h)
        if w>50 :
            if debug >= 2: print('폭이 너무 넓으면 패스')
            continue  #폭이 너무 넓으면 패스
        if ((x+w) > (wh_w - 20) and w<20) or ((x+w) > (wh_w - 10) and w<25):
            if debug >= 2: print('너무 우측에 위치하면 패스',x,w,wh_w)
            continue # 너무 우측에 위치하면 패스
        # 2 정도 떨어져 있고 합쳐지면 폭이 40이상 일 때
        cond3 = False
        if (x - (_x + _w)  ) <=2 and (x - (_x + _w)  ) >=0 :
            #합쳐질 영역 계산
            n_x = _x if x > _x else x
            n_x2 = _x + _w if x + w < _x + _w else x + w
            n_w = n_x2 - n_x
            cond3 = n_w < 40
            if debug >= 2: print('2정도 떨어져 있고 합치면 ',n_w, cond3)

        if (temp_cnt != 0) and (_x+_w > x or cond3)  :
            # 이전 박스와 새 박스가 겹칠 때
            n_x = _x if x > _x else x
            n_y = _y if y > _y else y
            n_x2 = _x + _w if x + w < _x + _w else x + w
            n_w = n_x2 - n_x
            n_y2 = _y + _h if y + h < _y + _h else y + h
            n_h = n_y2 - n_y
            if n_w==0 : n_w=0.001
            if debug >= 2: print('합치면? ', i, n_x, n_y, n_w, n_h, n_w * n_h, float(n_w) / n_h, '[', n_x, n_y, n_x + n_w, n_y + n_h, ']')  # 중복되어 새로 합쳐진 영역
            if n_w <= 35 and n_y > 5  and n_h < 45:  #10->5
                if debug >= 2: print('중복-합치기  ', i, n_x, n_y, n_w, n_h, n_w * n_h, float(n_w) / n_h, '[', n_x, n_y, n_x + n_w, n_y + n_h, ']')  # 중복되어 새로 합쳐진 영역
                box2[temp_cnt-1] = (n_x, n_y, n_w, n_h)
                _x, _y, _w, _h = n_x, n_y, n_w, n_h
            else: i-=1
        else:
            #이전 박스와 새 박스가 겹치지 않을 때
            rect_area = _w * _h  # area
            aspect_ratio = float(_w) / _h  # ratio = width/height
            cond1 = not((aspect_ratio >= 0.30) and (aspect_ratio <= 0.95) and (rect_area >= 500) and (rect_area <= 1420))
            # 숫자 1 전용 조건
            if debug >= 2: print('_h ',_h)
            cond2 = not(_h> 40 and _h< 50)
            if debug >= 2: print(cond1, cond2)
            if (temp_cnt != 0) and (cond1 and cond2) :
                if debug >= 2: print('이전 꺼 제거',_x, _y, _w, _h, '[', _x, _y, _x + _w, _y + _h, ']',rect_area,aspect_ratio)
                box2[temp_cnt - 1] = (x, y, w, h)
                temp_cnt -= 1
            else:
                rect_area = w * h  # area
                aspect_ratio = float(w) / h  # ratio = width/height
                cond1 = not ((aspect_ratio >= 0.35) and (aspect_ratio <= 0.9) and (rect_area >= 500) and (rect_area <= 1415))
                if debug >= 2: print(find_num-1, i, cond1)
                if find_num-1 == i and cond1: continue
                box2.append(box1[i])
            _x, _y, _w, _h = x, y, w, h
            temp_cnt += 1
            if debug >= 2: print(i,'[',temp_cnt,']', _x, _y, _w, _h, _w * _h, float(_w) / _h, '[', _x, _y, _x + _w, _y + _h, ']')


    #추출된 영역이  x축 90-125  y축 20-60 사이면 안 겹쳐도 합치기
    # 한글 영역 겹치면 합치기
    # _x, _y, _w, _h = 0, 0, 0, 0
    # temp_cnt = 0
    # for i in range(len(box1)):
    #     x, y, w, h =  box1[i][0], box1[i][1], box1[i][2], box1[i][3]
    #     if((x>=90 and (x+w) <=125 and y >= 17 and (y+h) <=60)):
    #         if temp_cnt==0 :
    #             box2.append(box1[i])
    #             _x, _y, _w, _h = x, y, w, h
    #             temp_cnt = 1
    #             print(i, _x, _y, _w, _h, _w * _h, float(_w) / _h,'[',_x, _y, _x+_w, _y+_h,']')
    #         else:
    #             n_x = _x  if x > _x else x
    #             n_y = _y  if y > _y else y
    #             n_x2 = _x+_w if x+w < _x+_w else x+w
    #             n_w = n_x2 - n_x
    #             n_y2 = _y+_h if y+h < _y+_h else y+h
    #             n_h = n_y2 - n_y
    #             print('hangul  ',i, n_x, n_y, n_w, n_h, n_w * n_h, float(n_w) / n_h,'[',n_x, n_y, n_x+n_w, n_y+n_h,']')   # 중복되어 새로 합쳐진 영역
    #             box2[0] =  (n_x, n_y, n_w, n_h)
    #             _x, _y, _w, _h = n_x, n_y, n_w, n_h
    # if temp_cnt !=0 : box1.append(box2[0])

    # #추출된 영역 정렬 - x축 기준
    # for i in range(len(box1)):  ##Buble Sort on python
    #     for j in range(len(box1) - (i + 1)):
    #         if box1[j][0] > box1[j + 1][0]:
    #             temp = box1[j]
    #             box1[j] = box1[j + 1]
    #             box1[j + 1] = temp

    # #추출된 영역 정렬 - x축 기준, 영역 겹치면 합치기
    # _x, _y, _w, _h = 0, 0, 0, 0
    # box2 = []
    # for i in range(len(box1)):
    #     x, y, w, h =  box1[i][0], box1[i][1], box1[i][2], box1[i][3]
    #     if not((w/h>=0.35)and(w/h<=0.9)and(w*h>=500)and(w*h<=1400)) : continue
    #     # print(box1[i][0], box1[i][1], box1[i][2], box1[i][3])
    #     # if (_x <= box1[i][0] and box1[i][0] <= _x+_w and _y <= box1[i][1] and box1[i][1] <= _y+_h ): #continue  # 겹치면 제거
    #     total_area = (w * h) + (_w * _h)
    #     include_area = getIntersection(x,y,w,h,_x,_y,_w,_h)
    #     # print('[',x,y,w,h,']','[',_x,_y,_w,_h,']',total_area, include_area, include_area/total_area)
    #     if (include_area!=0) or (_x <= box1[i][0] and box1[i][0] <= _x+_w and _y <= box1[i][1] and box1[i][1] <= _y+_h ) :  # continue  # 겹치면 제거
    #     # if (_x <= box1[i][0] and box1[i][0] <= _x+_w and _y <= box1[i][1] and box1[i][1] <= _y+_h ): #continue  # 겹치면 제거
    #         n_x = _x  if box1[i][0] > _x else box1[i][0]
    #         n_y = _y  if box1[i][1] > _y else box1[i][1]
    #         n_w = _w  if box1[i][2] < _w else box1[i][2]
    #         n_w += abs(box1[i][0] - _x)
    #         n_h = _h  if box1[i][3] < _h else box1[i][3]
    #         n_h += abs(box1[i][1] - _y)
    #         # print('new  ',n_x, n_y, n_w, n_h, n_w * n_h, float(n_w) / n_h)   # 중복되어 새로 합쳐진 영역
    #         box2[i-1] =  (n_x, n_y, n_w, n_h)
    #     else:
    #         box2.append(box1[i])
    #         _x, _y, _w, _h = box1[i][0], box1[i][1], box1[i][2], box1[i][3]
    #         # print(_x, _y, _w, _h, _w * _h, float(_w) / _h)
    result_num_count = len(box2)
    if debug >= 1: print(result_num_count)
    sum_y, sum_w, sum_h = 0, 0, 0
    # 평균 구하기
    for m in range(result_num_count):
        if (result_num_count==7 and m==2) : continue
        x, y, w, h = box2[m]
        sum_y += y
        sum_w += w
        sum_h += h
    if result_num_count<=0 :
        mean_y = round(sum_y)
        mean_w = round(sum_w)
        mean_h = round(sum_h)
    else:
        mean_y = round(sum_y/(result_num_count))
        mean_w = round(sum_w/(result_num_count))
        mean_h = round(sum_h/(result_num_count))

    #결과가 7보다 적을 때 : 빈 공간을 찾기
    each_space=[]
    o_x1, o_x2 = 0,0
    empty_pos = 0
    if result_num_count==6:
        for m in range(result_num_count):
            x, y, w, h = box2[m]
            x1=x
            x2=x+w
            space=x1 - o_x2
            o_x1, o_x2 = x1, x2
            each_space.append(space)
        if debug >= 2: print('각 글자간 간격',each_space)
        # 빈공간 위치 찾기
        max1, max1_pos, max2, max2_pos = 0,0,0,0
        for i in range(len(each_space)):
            if i==0 : continue  #맨 앞의 것은 무시
            if each_space[i] > max1:
                max1 = each_space[i]
                max1_pos = i
        for i in range(len(each_space)):
            if i==0 : continue  #맨 앞의 것은 무시
            if i==max1_pos : continue  #max1 무시
            if each_space[i] > max2:
                max2 = each_space[i]
                max2_pos = i
        if debug >= 2: print(max1, max2)
        if abs(max1 - max2) < 12 :
            empty_pos = max1_pos
        elif max1_pos==2 : empty_pos = 0
        else : empty_pos = 6
        if debug >= 2: print('빈곳 위치 ',empty_pos)
        #빈곳의 좌표 구하기
        if empty_pos==0:
            left_x = 0
            right_x =  box2[empty_pos][0]
        elif empty_pos==6:
            left_x = box2[empty_pos-1][0]+box2[empty_pos-1][2]
            right_x =  wh_w
        else:
            left_x = box2[empty_pos-1][0]+box2[empty_pos-1][2]
            right_x = box2[empty_pos][0]
        if empty_pos==0 or empty_pos==3 :  n_x = right_x-mean_w-3
        else: n_x = left_x+3
        n_w = mean_w
        n_y = mean_y
        n_h = mean_h-2
        if debug >= 2: print('예상 숫자 좌표', n_x, n_y, n_w, n_h,'   ',left_x,right_x)
        box2.insert(empty_pos,(n_x, n_y, n_w, n_h))
        result_num_count +=1

    result_num_str=''
    # 각 영역을 파일로 저장
    for m in range(result_num_count):
        x, y, w, h = box2[m]
        if debug >= 1 : print('final_1',m, x, y, w, h, '[', x, y, x + w, y + h, ']', w*h, w/h, mean_y, mean_w, mean_h,'diff',mean_y-y, w-mean_w, h-mean_h)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if m ==2:  #한글
            if (w - mean_w) < -2: w += (mean_w - w+2)
            if (h-mean_h) < -4 :
                h += (mean_h-h-3)
                if (mean_h-h)==0: y-= 0
                else : y -= round((mean_h-h)/2)
        else: #숫자
            if (mean_y-y) < -15 : y = mean_y
            if (mean_y-y) < -2 : y -= (y-mean_y)
            if (h-mean_h) < -15 : h = mean_h+9
            if (h-mean_h) < -2 : h += (mean_h-h)
            if debug >= 2: print(x, box2[m-1][0], box2[m-1][2], (box2[m-1][0]+box2[m-1][2]), x-(box2[m-1][0]+box2[m-1][2]), m)
            if (x-(box2[m-1][0]+box2[m-1][2]) > 7 ) and (m > 3) :
                if debug >= 2: print('x위치 이상 ',x-(box2[m-1][0]+box2[m-1][2]), m)
                # x -=(x-(box2[m-1][0]+box2[m-1][2]))-7
                x -= round((x - (box2[m - 1][0] + box2[m - 1][2]))/2)-3
            #1의 경우
            if (w-mean_w) < -2:
                w += (mean_w - w +3)
                if (mean_w - w+3)==0 : x -= 0
                else : x -= round((mean_w - w+3) / 2)
        if debug >= 1 : print('final_2        ',m, x, y, w, h, '[', x, y, x + w, y + h, ']', w*h, w/h, mean_y, mean_w, mean_h,'diff',mean_y-y, w-mean_w, h-mean_h)

        x_offset = 3 #8
        y_offset = 3
        if w/h < 0.45 : x_offset = 3+5  #1처럼 폭이 좁을 때 폭을 넓게 해서 저장
        if w/h > 0.8 and h<35 : y_offset = 3+2  #1처럼 폭이 좁을 때 폭을 넓게 해서 저장
        box2[m] = (x, y, w, h)
        # print(x, y, w, h,x_offset, y_offset)
        # print(box2[m][1]-y_offset, box2[m][1]+box2[m][3]+y_offset, box2[m][0]-x_offset, box2[m][0]+box2[m][2]+x_offset)
        # 좌우상하로 확대했을 때 0,0,wh_w,wh_h 안에 들어가게 x_offset,y_offset조정
        if (y-y_offset)<0 : y_offset -=  -(y-y_offset)
        if (x-x_offset)<0 : x_offset -=  -(x-x_offset)
        if (y+h+y_offset >= wh_h) : y_offset -=  -(wh_h-(y+y_offset))
        if (x+w+x_offset >= wh_w) : x_offset -=  -(wh_w-(x+x_offset))
        # print(x_offset,y_offset,box2[m][1]-y_offset, box2[m][1]+box2[m][3]+y_offset, box2[m][0]-x_offset, box2[m][0]+box2[m][2]+x_offset)

        # each_chr = copy_img[box2[m][1]-y_offset:box2[m][1]+box2[m][3]+y_offset,box2[m][0]-x_offset:box2[m][0]+box2[m][2]+x_offset]
        each_chr = copy_img[y-y_offset:y+h+y_offset,x-x_offset:x+w+x_offset]
        #수정된 영역(제거, 합치기 등)으로 박스 그리기
        cv2.rectangle(img, (x-x_offset, y-y_offset), (x + w+x_offset, y + h+y_offset), (0, 0, 255), 1)  #red line

        r_name = dst_path+'/'+fname[1] + '_'+str(m) + fext[1]
        if debug >= 1: print(r_name)
        cv2.imwrite(r_name, each_chr)  #파일로 저장
        if debug >= 5:
            cv2.imshow('6', img)
            cv2.waitKey(0)
        # 문자 판별 위해 저장된 각 영역그림 파일 로드하여 메모리에 저장
        test_data = img_utils.read_data_from_img_one(r_name,x_size,y_size)
        # print(test_data.shape)  # (1 ,35*25)

        # Get one and predict - 숫자 판별
        result = sess.run(tf.argmax(logits, 1), feed_dict={X: test_data, keep_prob: 1})
        print("step=%s" % m, result, type(result[0]))
        if m == 2:  # 한글
            result_num_str +='_'
        else:
            result_num_str +=str(result[0])

        if m==2: #한글
            num_index = 0
        else: #숫자
            num_index = int(result[0])

        n_name = '%s/%s_%04d%s' %(dst_path, result[0], num_cnt[num_index] ,fext[1])
        num_cnt[num_index] += 1
        if debug >= 1: print('extracted image name:', n_name)
        if os.path.isfile(n_name): os.remove(n_name)
        os.rename(r_name, n_name)

    # cv2.imshow(filename,img)
    # cv2.waitKey(0)
    return result_num_str


debug = 1 #0 final    #1 거의 결과  #2 후보 정하기 #3 박스 모든 후보를 차례로  #4 canny # 중간 이미지 보이기
num_cnt = [0]*11

# 숫자 인식 위한 준비
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, model_file_path)
vehicle_num_str = ExtractNumber(sess, image_file)
print(f'recognized vehicle number: {image_file} --> "{vehicle_num_str}"')
if debug == 1:
  cv2.imshow(image_file, cv2.imread(image_file, cv2.IMREAD_COLOR))
  cv2.waitKey(0)
sess.close()
