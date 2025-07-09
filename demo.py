from AIDetector_pytorch import Detector
import imutils
import cv2
import warnings

# 忽略torch.meshgrid警告
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

class VehicleCounter:
    def __init__(self):
        self.vehicle_count = 0
        self.counted_ids = set() 

    def update(self, detections):
        """更新计数"""
        current_ids = set()
        if 'list_of_ids' in detections:
            for obj_id in detections['list_of_ids']:
                current_ids.add(obj_id)
                if obj_id not in self.counted_ids:
                    self.vehicle_count += 1
                    self.counted_ids.add(obj_id)
        return self.vehicle_count

def main():
    name = 'demo'
    det = Detector()
    counter = VehicleCounter() 
    
    cap = cv2.VideoCapture('traffic_car.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    t = int(1000 / fps)
    print(f'视频帧率 (FPS): {fps}')

    videoWriter = None

    while True:
        _, im = cap.read()
        if im is None:
            break
        
        # 进行目标检测
        result = det.feedCap(im)
        print(f"检测到 {len(result['list_of_ids'])} 个目标, IDs: {result['list_of_ids']}")

        # 更新计数
        total_vehicles = counter.update(result)
        print(f"当前目标计数: {total_vehicles}")
        
        # 获取检测结果帧
        result_frame = result['frame']
        result_frame = imutils.resize(result_frame, height=500)
        
        # 在帧上显示计数
        cv2.putText(result_frame, f"Vehicles: {len(result['list_of_ids'])}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 初始化视频写入器
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result_frame.shape[1], result_frame.shape[0]))

        videoWriter.write(result_frame)
        cv2.imshow(name, result_frame)
        
        # 按ESC退出
        if cv2.waitKey(t) & 0xFF == 27:
            break

    cap.release()
    if videoWriter is not None:
        videoWriter.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成，总共检测到 {total_vehicles} 目标")

if __name__ == '__main__':
    main()