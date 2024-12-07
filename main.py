import sys
import yaml
import cv2
import os
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.core.detector import Detector

def load_config():
   try:
       with open('config/config.yaml', 'r', encoding='utf-8') as f:
           return yaml.safe_load(f)
   except:
       try:
           with open('config/config.yaml', 'r', encoding='cp1252') as f:
               return yaml.safe_load(f)
       except:
           with open('config/config.yaml', 'r') as f:
               return yaml.safe_load(f)

def create_directories():
   dirs = ['input/videos', 'output/videos', 'weights/yolo']
   for dir in dirs:
       os.makedirs(dir, exist_ok=True)

def main():
   try:
       # Set console encoding for Windows
       if sys.platform.startswith('win'):
           import ctypes
           ctypes.windll.kernel32.SetConsoleCP(65001)
           ctypes.windll.kernel32.SetConsoleOutputCP(65001)

       # Initialize app
       create_directories()
       app = QApplication(sys.argv)
       config = load_config()

       # Create main window
       window = MainWindow()
       window.show()

       # Start event loop
       sys.exit(app.exec())

   except Exception as e:
       print(f"Error: {str(e)}")
       sys.exit(1)

if __name__ == '__main__':
   main()