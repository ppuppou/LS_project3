import pandas as pd
import joblib
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive
from shiny.ui import update_slider, update_numeric, update_select, update_navs
import seaborn as sns
import pathlib
import plotly.express as px
from shinywidgets import render_plotly, output_widget
import numpy as np
import matplotlib
from sklearn.metrics import pairwise_distances
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt
import plotly.io as pio
import calendar
import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats




train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# train.to_csv('train_.csv', index = False, encoding = 'utf-8-sig')
# test.to_csv('test_.csv', index = False, encoding = 'utf-8-sig')




app_dir = pathlib.Path(__file__).parent
# ===== 한글 깨짐 방지 설정 =====
plt.rcParams["font.family"] = "Malgun Gothic"   # 윈도우: 맑은 고딕
plt.rcParams["axes.unicode_minus"] = False      # 마이너스 기호 깨짐 방지

# 폰트 파일 경로
APP_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(APP_DIR, "www", "fonts", "NanumGothic-Regular.ttf")

# 폰트 적용
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"  # Matplotlib
    print(f"✅ 한글 폰트 적용됨: {font_path}")
else:
    plt.rcParams["font.family"] = "sans-serif"
    print(f"⚠️ 한글 폰트 파일 없음 → {font_path}")

# Plotly 기본 폰트 설정
pio.templates["nanum"] = pio.templates["plotly_white"].update(
    layout_font=dict(family="NanumGothic")
)
pio.templates.default = "nanum"

