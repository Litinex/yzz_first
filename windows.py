import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import cv2
import torch
import numpy as np

class YOLOv5App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLOv5 GUI')
        self.setGeometry(100, 100, 1600, 900)  # 增大窗口尺寸

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.image_label = QLabel('Input Image/Video', self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(750, 750)  # 增大图片显示区域
        self.image_label.setStyleSheet("border: 2px solid #000;")

        self.result_label = QLabel('Result Image/Video', self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedSize(750, 750)  # 增大结果显示区域
        self.result_label.setStyleSheet("border: 2px solid #000;")

        self.upload_image_btn = QPushButton('Upload Image', self)
        self.upload_image_btn.setFixedSize(200, 50)
        self.upload_image_btn.setFont(QFont("Helvetica", 14, QFont.Bold))
        self.upload_image_btn.clicked.connect(self.upload_image)

        self.upload_video_btn = QPushButton('Upload Video', self)
        self.upload_video_btn.setFixedSize(200, 50)
        self.upload_video_btn.setFont(QFont("Helvetica", 14, QFont.Bold))
        self.upload_video_btn.clicked.connect(self.upload_video)

        # 美化按钮
        self.upload_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* 绿色背景 */
                color: white; /* 白色字体 */
                border: none;
                border-radius: 25px; /* 圆角 */
            }
            QPushButton:hover {
                background-color: #45a049; /* 悬停时颜色 */
            }
            QPushButton:pressed {
                background-color: #3e8e41; /* 按下时颜色 */
            }
        """)

        self.upload_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; /* 蓝色背景 */
                color: white; /* 白色字体 */
                border: none;
                border-radius: 25px; /* 圆角 */
            }
            QPushButton:hover {
                background-color: #1E88E5; /* 悬停时颜色 */
            }
            QPushButton:pressed {
                background-color: #1976D2; /* 按下时颜色 */
            }
        """)

        layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.result_label)
        button_layout.addStretch()
        button_layout.addWidget(self.upload_image_btn)
        button_layout.addWidget(self.upload_video_btn)
        button_layout.addStretch()

        layout.addLayout(image_layout)
        layout.addLayout(button_layout)

        central_widget.setLayout(layout)

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

        # Timer for updating video frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.video_path = None
        self.cap = None

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.display_image(file_name)
            self.process_image(file_name)

    def upload_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Videos (*.mp4 *.avi *.mov)", options=options)
        if file_name:
            self.video_path = file_name
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer.start(30)
            self.process_video_thread(file_name)

    def display_image(self, file_name):
        pixmap = QPixmap(file_name)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
        else:
            self.cap.release()
            self.timer.stop()

    def process_image(self, file_name):
        image = cv2.imread(file_name)
        results = self.model(image)
        processed_image = np.squeeze(results.render())  # YOLOv5 render

        processed_image_path = 'processed_image.jpg'
        cv2.imwrite(processed_image_path, processed_image)

        pixmap = QPixmap(processed_image_path)
        self.result_label.setPixmap(pixmap.scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

    def process_video_thread(self, file_name):
        self.thread = QThread()
        self.worker = VideoProcessor(file_name, self.model)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.update_frame_signal.connect(self.update_result_frame)
        self.thread.start()

    def update_result_frame(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.result_label.setPixmap(pixmap.scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

class VideoProcessor(QThread):
    finished = pyqtSignal()
    update_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, file_name, model):
        super().__init__()
        self.file_name = file_name
        self.model = model

    def run(self):
        cap = cv2.VideoCapture(self.file_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('processed_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results = self.model(frame)
                processed_frame = np.squeeze(results.render())  # YOLOv5 render
                out.write(processed_frame)
                self.update_frame_signal.emit(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            else:
                break

        cap.release()
        out.release()
        self.finished.emit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = YOLOv5App()
    main_window.show()
    sys.exit(app.exec_())
