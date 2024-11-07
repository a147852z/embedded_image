import cv2
import numpy as np
import json
from skimage.feature import local_binary_pattern
from skimage import exposure
"""
N: LBP取樣點數量。
stride: LBP的區域步伐大小。
LBP_RADIUS: LBP的半徑。
LBP_METHOD: LBP的方法。
similarity_threshold: 相似度的閾值。
road_patch_position: 用來定義道路特徵範本區域的起始位置。
road_patch_size: 道路特徵範本區域的大小。
"""
def load_config(config_path):
    """從JSON檔案載入參數配置"""
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_image(filepath):
    """讀取影像並轉換為灰階"""
    image = cv2.imread(filepath)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

def edge_detection(gray_image):
    """執行Sobel邊緣檢測"""
    edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
    return cv2.convertScaleAbs(edges)

def compute_lbp(gray_image, N, LBP_RADIUS, LBP_METHOD):
    """計算影像的LBP紋理特徵"""
    lbp = local_binary_pattern(gray_image, N, LBP_RADIUS, method=LBP_METHOD)
    lbp_hist = exposure.equalize_hist(lbp)  # optional for visibility
    return lbp, lbp_hist

def compute_histogram(patch):
    """計算給定區域的直方圖"""
    hist = cv2.calcHist([patch.astype(np.uint8)], [0], None, [256], [0, 256])
    return hist

def find_similar_regions(lbp, road_hist, stride, similarity_threshold):
    """根據LBP直方圖搜尋相似區域"""
    similar_regions = np.zeros_like(lbp, dtype=np.uint8)
    for i in range(0, lbp.shape[0] - stride, stride):
        for j in range(0, lbp.shape[1] - stride, stride):
            patch = lbp[i:i+stride, j:j+stride]
            patch_hist = compute_histogram(patch)
            similarity = cv2.compareHist(road_hist, patch_hist, cv2.HISTCMP_CORREL)
            if similarity > similarity_threshold:
                similar_regions[i:i+stride, j:j+stride] = 255
    return similar_regions

def apply_color_mask(image, mask):
    """將相似區域用掩膜進行填色"""
    return cv2.bitwise_and(image, image, mask=mask)

def display_results(images, titles):
    """顯示影像結果"""
    for title, img in zip(titles, images):
        cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_road_patch(lbp, config):
    """取得道路的特徵區域，並計算其直方圖"""
    size = config["road_patch_size"]
    height, width = lbp.shape
    print(height, width)
    if config["road_patch_position"] == "center":
        # 計算影像中心的區域
        x_start, y_start = height // 2 - size // 2, width // 2 - size // 2
    elif config["road_patch_position"] == "center_bottom":
        # 計算影像中間偏下的區域
        x_start = height * 1 - size // 2  # 將位置設置為影像高度的3/4處
        y_start = width // 2 - size // 2       # 中間的水平位置
    else:
        # 使用JSON中提供的固定位置
        x_start, y_start = config["road_patch_position"]
    print(x_start, y_start)
    # 擷取道路區域的LBP特徵
    road_patch = lbp[x_start:x_start + size, y_start:y_start + size]
    road_hist = compute_histogram(road_patch)
    return road_hist

def main(image_path, config_path):
    # 載入配置
    config = load_config(config_path)[0]
    
    # 1. 讀取影像
    image, gray_image = load_image(image_path)
    
    # 2. 邊緣檢測
    edges = edge_detection(gray_image)
    
    # 3. LBP紋路統計
    lbp, lbp_hist = compute_lbp(gray_image, config["N"], config["LBP_RADIUS"], config["LBP_METHOD"])
    
    # 4. 定義道路的特徵區域，計算其直方圖
    road_hist = get_road_patch(lbp, config)
    
    # 5. 相似度搜尋
    similar_regions = find_similar_regions(lbp, road_hist, config["stride"], config["similarity_threshold"])
    
    # 6. 塗色
    output_image = apply_color_mask(image, similar_regions)
    
    # 7. 顯示結果
    # display_results(
    #     [image, edges, lbp.astype(np.uint8), similar_regions, output_image],
    #     ['Original Image', 'Edge Detection', 'LBP', 'Similar Regions', 'Segmented Road']
    # )
    display_results(
        [output_image],
        ['Segmented Road']
    )

# 執行主程式
main('test.jpg', 'setting.json')
