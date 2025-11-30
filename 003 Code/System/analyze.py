# -*- coding: utf-8 -*-
print("1/6: 기본 라이브러리 로딩", flush=True)
import sys
import os
import time

# 이제 에러가 나면 MFC 화면에 에러 내용이 뜰 것입니다.
sys.stderr = sys.stdout

import warnings
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 화면 표시 안 함 모드 설정
from matplotlib import pyplot as plt
import cv2
# 여기가 가장 무거운 구간입니다.
print("2/6: PyTorch 엔진 로딩 중", flush=True)
import torch
print("3/6: Segmentation 모델 로딩 중...", flush=True)
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

print("4/6: 사용자 정의 모듈 로딩...", flush=True)
# definition 안에서도 무거운 import가 있다면 여기서 멈출 수 있습니다.
import definition



if __name__ == "__main__":
    # sys.stdout.reconfigure(encoding='utf-8')
    # sys.argv[1:]로 스크립트명 이후의 인자만 추출
    image_pathes = sys.argv[1:]  # 이미지 폴더 경로
    # print(image_pathes)
    definition.seed_everything(42)
    warnings.filterwarnings("ignore")
    print("5/6: 경로 설정 및 데이터 준비", flush=True)
    image_path_list = image_pathes[0].split('\\')[:-1]  # 첫 번째 경로 사용
    image_dir = '\\'.join(image_path_list)
    image_file = [image.split("\\")[-1] for image in image_pathes]

    current_path = os.getcwd()  # 현재 위치
    # ckpt_path = os.path.join(current_path,'ckpt')
    ckpt_path = os.path.abspath(os.path.join(current_path, '..', "ckpt"))  # 모델 체크포인트 폴더
    
    print(ckpt_path, flush=True)
    # print(image_dir)
    # print(image_file)

    sample_path = os.path.abspath(os.path.join(current_path, '..', '..', "results"))  # 결과 저장 폴더
    sample_crop = os.path.join(sample_path,'preprocessing','Crop')  # 자른 이미지
    sample_label = os.path.join(sample_path,'preprocessing','Label')  # 자른 라벨

    sample_prediction = os.path.join(sample_path,'prediction')  # 결과 데이터
    sample_segmentation = os.path.join(sample_path,'segmentation')  # 세그멘테이션 결과 저장 폴더
    
    pd.set_option('display.max_colwidth', 5500)

    base_name = ['.'.join(x.split('.')[:-1]) for x in image_file]

    df = pd.DataFrame({
        'base_names': base_name,
        'file_name': image_file,
        'file_dir': [os.path.join(image_dir, x) for x in image_file],
        'autolabeling_dir': [os.path.join(sample_label, f'{x}_crop') for x in base_name],  # 라벨링 결과
        'img_dir': [os.path.join(sample_prediction, f'{x}_result') for x in base_name]})  # 결과 데이터
    
    definition.rebuild_dir(sample_crop)
    definition.build_dir(sample_label)
    definition.build_dir(sample_prediction)
    definition.build_dir(sample_segmentation)
    definition.cropping_image(df, sample_crop)  # 이미지 자르기: 512*512

    ENCODER = 'efficientnet-b2'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['cell']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # BATCH_SIZE = 8
    # create segmentation model with pretrained encoder
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # print(DEVICE)

    # Create test dataset
    test_dataset = definition.InferenceDataset(
        images_dir=image_dir,
        images_files=image_file,
        classes=CLASSES,
        preprocessing= definition.get_preprocessing(preprocessing_fn),
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=8, # 한 번에 하나의 원본 이미지(의 타일들)를 처리
        shuffle=False, 
        num_workers = 0,  # num_workers/는 시스템의 CPU 코어 수를 활용할 수 있지만, 여기서는 0으로 설정하여 단일 프로세스에서 실행
        # num_workers=os.cpu_count(), # 시스템의 CPU 코어 수 활용
        collate_fn=definition.inference_collate_fn
    )
    
    # 모델 불러오기
    # print("AI 모델을 메모리에 적재하는 중...", flush=True)
    # model.load_state_dict(torch.load(os.path.join(ckpt_path, 'Full_Image_weights.pth'), map_location=DEVICE))
    # best_model = model.eval().to(DEVICE)
    # print("AI 모델이 메모리에 적재되었습니다.", flush=True)
    # [변경] JIT 모델 로드
    print("6/6: 최적화된 AI 모델 로딩 중", flush=True)
    try:
        best_model = torch.jit.load(os.path.join(ckpt_path, 'Full_Image_weights_jit.pt'), map_location=DEVICE)
        best_model.eval()
        print("모델 준비 완료! 분석 시작", flush=True)
        
        
    except Exception as e:
        print(f"Error: 모델 로드 중 오류 발생: {e}", flush=True)
        sys.exit(1)

    @torch.no_grad() # 추론 시에는 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 속도를 높입니다.
    def inference_on_folder(model, dataloader, device, inference_dir):
        print("폴더 내 이미지에 대해 추론을 시작합니다...", flush=True)
        progress_bar = tqdm(dataloader, desc="Inference Progress", file=sys.stdout)
        
        full_masks = {}
        num = 0
        
        for _, data in enumerate(progress_bar):
            tiles_batch, h_starts, w_starts, heights, widths, image_names = data
            #print(f"\n[{i+1}/{len(dataloader)}] 배치 추론 중...")        
            tiles_batch = (tiles_batch).to(device)  
            
            # 모델 예측
            prediction = model(tiles_batch) # 출력: (batch_size, num_classes, H, W)
            predictions = prediction.cpu().numpy() 

            
            
            for j in range(len(image_names)):
                img_name = image_names[j]
                h_start = h_starts[j]
                w_start = w_starts[j]
                height = heights[j]
                width = widths[j]
                predicted_mask_tile = predictions[j]
                # 활성화 함수 적용 후 CPU로 이동하고 NumPy 배열로 변환
                # sigmoid 활성화 함수를 사용하면 [0, 1] 범위의 확률 맵이 출력됩니다.
                # 'cell' 클래스 하나만 예측하므로 prediction.squeeze()를 사용합니다.
                # threshold를 적용하여 이진 마스크로 변환 (0.5 이상이면 1, 아니면 0)
                
                # 단일 클래스 이진 세그멘테이션의 경우 (1, H, W) -> (H, W)
                if len(CLASSES) == 1:
                    predicted_mask_tile = predicted_mask_tile.squeeze(0) # (H, W)
                    predicted_mask_tile = (predicted_mask_tile > 0.5).astype(np.uint8) * 255 # 이진 마스크로 변환 (0 또는 255)
                else:
                    # 다중 클래스 세그멘테이션의 경우 argmax 사용
                    predicted_mask_tile = np.argmax(predicted_mask_tile, axis=0).astype(np.uint8) # (H, W)
                    # 클래스 ID를 실제 픽셀 값으로 매핑해야 할 수 있습니다. (예: 0, 1, 2... 에 따라 다른 색상)
                    # 여기서는 예시로 255로 스케일링하여 저장합니다.
                    predicted_mask_tile = predicted_mask_tile * (255 // (len(CLASSES) -1)) if len(CLASSES) > 1 else predicted_mask_tile * 255
                tile_path = os.path.join(inference_dir,f"{img_name}_crop", f"{num:05d}.png")
                cv2.imwrite(tile_path, predicted_mask_tile)     
                
                if definition.has_less_white_pixels(tile_path):
                    os.remove(tile_path)
                    continue
                num += 1

                # --- 전체 마스크 재구성 로직 ---
                # 현재 이미지에 대한 전체 마스크가 딕셔너리에 없으면 초기화
                if img_name not in full_masks:
                    full_masks[img_name] = np.zeros((height, width), dtype=np.uint8)
                
                h_end = min(h_start + 512, height)
                w_end = min(w_start + 512, width)
                
                # 재구성할 마스크의 실제 높이와 너비
                actual_reconstruct_height = h_end - h_start
                actual_reconstruct_width = w_end - w_start
                
                # 예측된 타일 마스크에서 원본 영역에 해당하는 부분만 추출
                reconstructed_tile = predicted_mask_tile[:actual_reconstruct_height, :actual_reconstruct_width]
                
                # 전체 마스크에 재구성
                full_masks[img_name][h_start:h_end, w_start:w_end] = reconstructed_tile
            # 현재 배치의 추론 진행 상황 출력
            #print(f"[{i+1}/{len(dataloader)}] 배치 추론 완료.")

        # 모든 배치가 처리된 후, 최종 마스크들을 파일로 저장
        return full_masks
        
    # 추론 실행
    full_masks = inference_on_folder(best_model, test_dataloader, DEVICE, sample_label)
    
    idx = 0
    
    for img_name,full_mask in tqdm(full_masks.items()):
        # print(df.loc[idx,'file_name'])
        name = df.loc[idx,'base_names']
        storage_path = df.loc[idx, 'img_dir']
        definition.rebuild_dir(storage_path)
        
        img = cv2.imread(df.loc[idx,'file_dir'], cv2.IMREAD_COLOR_RGB)  # 원본 이미지
        img_blurrd, full_contour = definition.blurrd_img(df.loc[idx,'file_dir'])  # 블러 처리된 이미지
        prediction_img = definition.draw_contours_on_image(img_blurrd, full_mask, color=(0, 255, 0), full_mask=full_contour)  # contour 이미지
        total, hist_count, bin_edge = definition.analyze_segmentation_mask(full_mask, bin_size=100)
        definition.plot_histogram_skip_empty(hist_count, bin_edge, storage_path, name)

        idx += 1
        cv2.imwrite(os.path.join(storage_path, f"{name}_original.png"), img)
        cv2.imwrite(os.path.join(storage_path, f"{name}_blurrd.png"), img_blurrd)
        cv2.imwrite(os.path.join(sample_segmentation, f"{name}_mask.png"), full_mask)
        cv2.imwrite(os.path.join(storage_path, f"{name}_contour.png"), prediction_img)

    # count = len(os.listdir(sample_prediction))
    count = len(image_file)
    print(f"{count}/{sample_prediction}")
    sys.exit(0)