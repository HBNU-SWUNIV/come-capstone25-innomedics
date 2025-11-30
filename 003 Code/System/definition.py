# -*- coding: utf-8 -*-
from torch.utils.data import Dataset as BaseDataset
import torch
import albumentations as albu

import segmentation_models_pytorch as smp
from PIL import Image

import numpy as np
import pandas as pd 

import shutil

import os
import random
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def build_dir(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
def rebuild_dir(target_path):
    """폴더가 존재하면 삭제하고 새로 생성합니다."""
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path)

def cropping_image(df, save_path):
    size = 512
    for idx in df.index:
        base_names = df.loc[idx,'base_names']    
        img_path = df.loc[idx,'file_dir']

        img_rgb=cv2.imread(img_path, cv2.IMREAD_COLOR)  # 원본이미지
        max_height,max_width = img_rgb.shape[0], img_rgb.shape[1]

        num=0

        label_path = df.loc[idx,'autolabeling_dir']
        result_path = df.loc[idx,'img_dir']

        rebuild_dir(label_path)    
        build_dir(result_path)
        
        for height in tqdm(range(0, img_rgb.shape[0], size)):
            for width in range(0, img_rgb.shape[1], size):

                img_rgb_crop = img_rgb[height:height+size, width:width+size, :]  # 512*512 절삭(높이, 너비)
                if height+size > max_height or width+size > max_width:
                    pass
                else:
                    name = str(base_names)+'_Cropped_'+'%05d.png' % num
                    tile_dir = os.path.join(save_path, name)
                    cv2.imwrite(tile_dir,img_rgb_crop)
                    num += 1
            
class InferenceDataset(BaseDataset):
    CLASSES = ['x' for x in range(255)]
    CLASSES.append('cell')

    def __init__(
        self,
        images_dir,
        images_files,
        classes=None,
        preprocessing=None,
    ):
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in (images_files)]  # image_path
        self.preprocessing = preprocessing
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.tile_size = 512

        # 전체 이미지에서 모든 타일 정보를 미리 생성
        self.tiles_info = []
        for img_idx, image_path in enumerate(self.images_fps):
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            for h in range(0, height, self.tile_size):
                for w in range(0, width, self.tile_size):
                    self.tiles_info.append({
                        'img_idx': img_idx, # 원본 이미지 인덱스
                        'h_start': h,
                        'w_start': w,
                    })

    def __getitem__(self, i):
        # i번째 타일 정보 가져오기
        tile_info = self.tiles_info[i]
        img_idx = tile_info['img_idx']
        h_start = tile_info['h_start']
        w_start = tile_info['w_start']
        
        # 원본 이미지 로드
        image_path = self.images_fps[img_idx]
        image = cv2.imread(image_path)
        # 흑백일 때(2D array) BGR로 바꿔 주기
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            # 컬러 파일이라면 기본 BGR 로드 → RGB 로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        # 타일 크롭
        tile = image[h_start:h_start+self.tile_size, w_start:w_start+self.tile_size]
        
        # 전처리 적용
        if self.preprocessing:
            sample = self.preprocessing(image=tile)
            tile = sample['image']

        # 원본 이미지의 이름과 크기 정보 반환
        original_image_name = os.path.splitext(os.path.basename(image_path))[0]    
        original_height, original_width, _ = image.shape
        
        return tile, h_start, w_start, original_height, original_width, original_image_name

    def __len__(self):
        return len(self.tiles_info)
    
def to_tensor(x, **kwargs):
    if x.ndim == 2:  # 채널이 없는 경우    
        x = np.expand_dims(x, axis=-1)  # (H, W) → (H, W, 1)
    if x.shape[2] == 1:  # 1채널일 경우 → 3채널로 반복
        x = np.repeat(x, 3, axis=2)
    x = x.transpose(2, 0, 1).astype('float32')  # (C, H, W)
    return torch.from_numpy(x)
def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.PadIfNeeded(512, 512, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)
def inference_collate_fn(batch):
    """
    batch: [(tile, h, w, H, W, name),  … ]  형태의 리스트
    -> ([tiles…], [h_starts…], [w_starts…], [heights…], [widths…], [names…])
    """
    tiles, h_starts, w_starts, heights, widths, names = zip(*batch)
    tiles_batch = torch.stack(tiles)
    # tiles: tuple of tensors, h_starts 등은 tuple of int, names는 tuple of str
    return (tiles_batch), list(h_starts), list(w_starts), list(heights), list(widths), list(names)

def draw_contours_on_image(image, mask, color, full_mask=None):
    """Mask의 테두리를 주어진 색상으로 이미지에 그립니다."""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy()
    filtered_contours = []
    
    if full_mask is not None:
        # 2. 필터링 로직: full_mask 영역 안에 있는 contour만 선택
        for cnt in contours:
            try:
                # 3. 예측 contour의 중심점 계산
                M = cv2.moments(cnt)
                if M['m00'] == 0: continue # 면적이 0인 contour 방지
                
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # (배열 인덱싱은 y, x 순서)
                if full_mask[cy, cx] != 0:
                    filtered_contours.append(cnt)
            except Exception as e:
                # 중심점 계산이 불가능한 아주 작은 contour 등 예외 처리
                pass 
    else:
        # full_mask가 없으면 모든 contour를 그림
        filtered_contours = contours
    
    cv2.drawContours(image_with_contours, filtered_contours, -1, color, 10)
    return image_with_contours
def count_pixels_in_mask(mask):
    """마스크 안의 픽셀 수를 반환합니다."""
    return np.sum(mask)

def has_less_white_pixels(image_path, threshold = 1):
    """
    이미지 내에 지정된 threshold 값 미만의 흰색 픽셀이 있는지 확인합니다.
    :param image_path: 이미지 파일의 경로
    :param threshold: 흰색 픽셀의 최대 수
    :return: 지정된 threshold 값을 미만의 흰색 픽셀이 있으면 True, 그렇지 않으면 False
    """
    white_pixel_count = 0
    try:
        with Image.open(image_path) as img:
            # 이미지 모드 확인: 'L' 모드는 흑백, 'RGB'는 컬러
            if img.mode != 'L':
                # 흑백이 아니면 흑백으로 변환하여 픽셀 값 비교를 용이하게 합니다.
                img = img.convert('L')
            
            pixels = img.getdata()
            
            # 각 픽셀 값이 255(흰색)인지 직접 확인
            for pixel in pixels:
                if pixel == 255:
                    white_pixel_count += 1
                if white_pixel_count >= threshold:
                    return False
        return True
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        return False # 오류 발생 시 False 반환

def blurrd_img(tiff_files):
    ## 원본 이미지
    img = cv2.imread(tiff_files, cv2.IMREAD_COLOR_RGB)

    ## 블러 처리
    blur = cv2.medianBlur(img, 9)
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    ## hsv 채널분리
    h, s, v = cv2.split(hsv_img)
    _, thr_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    __cached__, thr_v = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    thr_v -= 255
    

    # Contours 찾기
    contours_s, _ = cv2.findContours(thr_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_v, _ = cv2.findContours(thr_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img_s = np.zeros_like(thr_s)
    cn_filled_s = cv2.drawContours(contour_img_s, contours_s, -1, (255,255,255), -1)

    contour_img_v = np.zeros_like(thr_v)
    cn_filled_v = cv2.drawContours(contour_img_v, contours_v, -1, (255,255,255), -1)

    ## hsv 채널 중 s,v 채널 영역 합침
    mask_area = cv2.bitwise_and(cn_filled_s, cn_filled_v)

    # Contours 찾기
    contours, _ = cv2.findContours(mask_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contours를 그릴 이미지 생성
    contour_img = np.zeros_like(mask_area)

    # Contours 그리기
    cn_filled = cv2.drawContours(contour_img, contours, -1, (255,255,255), -1)

    result_img = img.copy()
    result_img[cn_filled == 0] = [255, 255, 255]  # 바탕 흰색으로 채우기

    return result_img, mask_area

def analyze_segmentation_mask(mask_path, bin_size=1000):
    """
    저장된 전체 마스크 이미지를 분석하여 Contour 개수와 크기 분포를 반환합니다.

    :param mask_path: 분석할 마스크 이미지 파일 경로 (예: '..._mask.png')
    :param bin_size: 히스토그램의 구간 크기 (기본값: 1000 픽셀)
    :return: (전체 Contour 개수, 히스토그램 카운트, 히스토그램 구간) 튜플
    """
    
    # 1. 마스크 이미지를 그레이스케일로 로드
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask_image is None:
        print(f"오류: 이미지를 로드할 수 없습니다. 경로: {mask_path}")
        return 0, None, None

    # 2. Contour 찾기
    # cv2.RETR_EXTERNAL: 가장 바깥쪽의 Contour만 찾습니다.
    # cv2.CHAIN_APPROX_SIMPLE: Contour의 꼭짓점만 저장하여 메모리를 절약합니다.
    contours, _ = cv2.findContours(
        mask_image, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    total_contours = len(contours)
    # print(f"--- 마스크 분석 결과 ('{os.path.basename(mask_path)}') ---")
    # print(f"✅ 전체 Contour 개수: {total_contours} 개")
    
    if total_contours == 0:
        print("분석할 Contour가 없습니다.")
        return 0, None, None

    # 3. 각 Contour의 픽셀 면적(Area) 계산
    contour_areas = [cv2.contourArea(cnt) for cnt in contours]
    
    # 4. 1000 단위로 찢어서 히스토그램화
    max_area = max(contour_areas)
    
    # 히스토그램 구간(bins) 설정
    # 예: max_area가 3500이면, bins = [0, 1000, 2000, 3000, 4000]
    bins = np.arange(0, max_area + bin_size, bin_size)
    
    # NumPy를 사용하여 히스토그램 계산
    hist_counts, bin_edges = np.histogram(contour_areas, bins=bins)
    
    # # 5. 결과 출력
    # print("\n--- Contour 크기 분포 (히스토그램) ---")
    # for i in range((total_contours)):
    #     print(f"{i}번째 contour 크기: {contour_areas[i]} 픽셀")
        
    return total_contours, hist_counts, bin_edges
def plot_histogram_skip_empty(hist_counts, bin_edges, save_path ,name):
    """
    계산된 히스토그램 데이터에서 카운트가 0인 구간을 '건너뛰고' (제외하고)
    막대그래프로 시각화하고 저장합니다.
    """
    
    if hist_counts is None or np.sum(hist_counts) == 0:
        print("시각화할 히스토그램 데이터가 없습니다.")
        return

    # --- 수정된 부분: 10이하인 구간 필터링 ---
    
    # 1. 카운트가 0보다 큰(> 0) 구간의 인덱스를 찾습니다.
    non_zero_indices = np.where(hist_counts > 10)[0]
    
    if len(non_zero_indices) == 0:
        print("시각화할 데이터가 없습니다 (모든 구간의 카운트가 0입니다).")
        return

    # 2. 0이 아닌 카운트만 필터링합니다.
    filtered_counts = hist_counts[non_zero_indices]
    
    # 3. 필터링된 구간에 해당하는 Bin 레이블을 생성합니다.
    # 예: "0-1000", "2000-3000" (1000-2000이 비어있다면 건너뜀)
    bin_labels = []
    for i in non_zero_indices:
        label = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
        bin_labels.append(label)
    # --- 여기까지 수정 ---
    plt.savefig(os.path.join(save_path,f'{name}_histogram.png'))
    # # 4. 필터링된 데이터로 막대그래프를 생성합니다.
    # # 그래프의 너비를 데이터 개수에 따라 동적으로 조절
    # plt.figure(figsize=(max(10, len(bin_labels) * 0.5), 6))
    # plt.bar(bin_labels, filtered_counts, color='skyblue', edgecolor='black')
    
    # plt.title('Histogram of Contour Areas (10 more section)')
    # plt.xlabel('Pixel Area Range (px)')
    # plt.ylabel('Number of Contours (Count)')
    # # plt.xticks(rotation=45, ha='right') # 레이블이 겹치지 않게 회전
    # plt.tight_layout() # 레이아웃 최적화
    
    
    # plt.show()
    # print(f"\n✅ (카운트가 10미만인 구간 제외) 히스토그램 그래프가 '{save_path}'에 저장되었습니다.")

