import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fontPath = "./static/Fonts/NIAGENG.TTF"
fontName = fm.FontProperties(fname=fontPath, size=12).get_name()
plt.rc("font", family=fontName)
mpl.rcParams["axes.unicode_minus"] = False
from django_plotly_dash import DjangoDash
import warnings; warnings.filterwarnings('ignore')

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost
import lightgbm
import os 

datafile_name = "final_pivot_df48.xlsx"
PATH = os.path.dirname(os.path.abspath(__file__)) + '/' +datafile_name
data = pd.read_excel(PATH)
print(f"불러온 data shape: {data.shape}")


####################################################함수정의때 사용
#제거할 컬럼
dropcols = ["진료구분", "target", "키", "gadab", "c/i_ratio(식후)", "인슐린외처방내역(glp1-ra)", "지속형+GLP1-RA사용여부", 'egfr', 'c/i_ratio(식전)', 'glucose(식후)', 'bc_ratio', 'ast_alt_ratio']
#인적정보 컬럼
humancols = ["나이", "몸무게", "성별", "bmi"]
#약정보 컬럼
medcols = ['a-glucosidaseinhibitor', 'dppiv', 'meglitinide', 'metformin', 'sglt2i', 'su', 'tzd', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)']
#검사정보 컬럼
checkcols = ['alt', 'ast', 'bun', 'cr', 'cr(urine)', 'crp', 'glucose(식전)','hba1c', 'hdl', 'ica', 'ketone(urine)', 'ldl', 'r-gtp', 'tc', 'tg', 'cpep핵의학(식전)', 'cpep핵의학(식후)', 'insulin핵의학(식전)', 'insulin핵의학(식후)']
usesc = humancols + medcols + checkcols
print(usesc)

#사용모델
DecisionTree = DecisionTreeRegressor(random_state=0)
RandomForest = RandomForestRegressor(random_state=0)
XGBoost = xgboost.XGBRegressor(random_state=0)
LightGBM = lightgbm.LGBMRegressor(random_state=0)
models = [DecisionTree, RandomForest, XGBoost, LightGBM]


#################################################원본데이터 
#원래 컬럼
orgcols = ['나이', '몸무게', '성별', '진료구분', '키', 'alt', 'ast', 'bun', 'cr', 'cr(urine)', 'crp', 'gadab', 'glucose(식전)', 'hba1c', 'hdl', 'ica', 'ketone(urine)', 'ldl', 'r-gtp', 'tc', 'tg', 'a-glucosidaseinhibitor', 'dppiv', 'meglitinide', 'metformin', 'sglt2i',
       'su', 'tzd', '인슐린외처방내역(glp1-ra)', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)', '지속형+GLP1-RA사용여부', 'cpep핵의학(식전)', 'cpep핵의학(식후)', 'insulin핵의학(식전)', 'insulin핵의학(식후)', 'glucose(식후)', 'bmi']
orgdf = data.copy()
orgdf = data[orgcols]
###성별한글변환
orgdf["성별"] = orgdf["성별"].replace({0:"여성", 1:"남성"})
###케톤한글변환
orgdf["ketone(urine)"] = orgdf["ketone(urine)"].replace({0:"Negative", 1:"Trace", 2:"Small", 3:"Moderate", 4:"Large"})
###ica한글변환
orgdf["ica"] = orgdf["ica"].replace({0:"없음", 1:"있음"})
###약종류한글변환
orgdf[['a-glucosidaseinhibitor', 'dppiv', 'meglitinide', 'metformin', 'sglt2i',
       'su', 'tzd', '인슐린외처방내역(glp1-ra)', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)', '지속형+GLP1-RA사용여부']] = orgdf[['a-glucosidaseinhibitor', 'dppiv', 'meglitinide', 'metformin', 'sglt2i',
       'su', 'tzd', '인슐린외처방내역(glp1-ra)', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)', '지속형+GLP1-RA사용여부']].replace({0:"미처방", 1: "처방"})
print(f"원본 데이터 프레임 shape: {orgdf.shape}")



#################################################대시보드 구성

colors = {
    'background': '#d2d2d2',
    'text': '#000000'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DjangoDash('dist', external_stylesheets=external_stylesheets)
app.layout = html.Div([

    ####################### 1단
    html.Div([
        ###변수별 분포
        # html.H4("변수별 분포"),
        html.Div([
            #드롭박스1: 변수명
            html.H4("변수별분포"),
            html.Div(
                dcc.Dropdown(
                    orgcols,
                    "나이",
                    id="xaxis_histogram"
                ), style={"width": "49%", "display":"inline-block"}
            ),
            #드롭박스2: 구분명
            html.Div(
                dcc.Dropdown(
                    ["성별", "ica", "ketone(urine)", 'a-glucosidaseinhibitor', 'dppiv','meglitinide', 'metformin', 'sglt2i', 'su', 'tzd', '인슐린외처방내역(glp1-ra)', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)', '지속형+GLP1-RA사용여부'],
                    "성별",
                    id="histogram_color"
                ), style={"width":"49%", "float":"right", "display":"inline-block"}
            ),
            #히스토그램 그래프
            html.Div(
                dcc.Graph(id="histogram")
            )
        ], style={"width":"35%", "padding": "20px"}),


        ###상관관계        
        html.Div([
            html.H4("변수별 상관관계"),
            #체크박스: 변수명
            html.Div(
                dcc.Checklist(
                    ['몸무게', '키', 'alt', 'ast', 'bun', 'cr', 'cr(urine)', 'crp', 'gadab', 'glucose(식전)', 'hba1c', 'hdl', 'ldl', 'r-gtp', 'tc', 'tg', 'cpep핵의학(식전)', 'cpep핵의학(식후)', 'insulin핵의학(식전)', 'insulin핵의학(식후)', 'glucose(식후)', 'bmi'],
                    ["몸무게", "키"],
                    id="pair_checklist",
                    labelStyle={"display":"inline-block"},
                    style={"width":"70%"}
                )
            ),
            #드롭다운: 구분명            
            html.Br(),
            html.Div(
                dcc.Dropdown(
                    ["성별", "ica", "ketone(urine)", 'a-glucosidaseinhibitor', 'dppiv','meglitinide', 'metformin', 'sglt2i', 'su', 'tzd', '인슐린외처방내역(glp1-ra)', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)', '지속형+GLP1-RA사용여부'],
                    "성별",
                    id="pair_color"
                )
            )
        ], style={"width":"15%", "padding":"20px"}),

        #상관관계 그래프
        html.Div(
            dcc.Graph(id="pairplot", figure={"layout":{"width":800, "height":550}})
        )
    ], style={"display":"flex", "flex-direction":"row"}),
])


#변수분포
@app.callback(
    Output("histogram", "figure"),
    Input("xaxis_histogram", "value"),
    Input("histogram_color", "value")
    )
def get_histogram(xaxis_histogram, histogram_color):
    fig = px.histogram(orgdf, x=xaxis_histogram, color=histogram_color, marginal="box")
    return fig

#상관관계
@app.callback(
    Output("pairplot", "figure"),
    Input("pair_checklist", "value"),
    Input("pair_color", "value"))
def get_pairplot(pair_checlist, pair_color):
    fig = px.scatter_matrix(orgdf, dimensions=pair_checlist, color=pair_color)    
    return fig


# if __name__ == "__main__":
#     app.run_server(debug=True)