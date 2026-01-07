import cv2
import json

# 全局变量
points = []
img_display = None

def click_event(event, x, y, flags, param):
    global points, img_display
    
    # 监听鼠标左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
        # 记录坐标
        points.append((x, y))
        
        cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Node Extractor", img_display)
        print(f"Captured Node: ({x}, {y})")

def extract_nodes(image_path, output_json="real_nodes.json"):
    global img_display
    
    # 读取原始图片
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found!")
        return

    
    img_display = img.copy()
    cv2.imshow("Node Extractor", img_display)

    # 设置鼠标回调
    cv2.setMouseCallback("Node Extractor", click_event)

    print("=== 操作指南 ===")
    print("1. 请用鼠标左键点击所有【路口】和【死胡同】。")
    print("2. 不用点得非常准，大概在路中间就行（后续会有算法自动吸附对齐）。")
    print("3. 点完后，按键盘 's' 保存并退出。")
    print("4. 按 'q' 不保存直接退出。")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 保存数据
            with open(output_json, 'w') as f:
                json.dump(points, f)
            print(f"Success! {len(points)} nodes saved to {output_json}")
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_nodes("standard_reference_map.png", output_json="real_nodes_corrected.json")