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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost
import lightgbm
import os 

datafile_name = "final_pivot_df48.xlsx"
PATH = os.path.dirname(os.path.abspath(__file__)) + '/' +datafile_name
data = pd.read_excel(PATH)
print(f"dist.py 갱신중")


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
# print(usesc)

#사용모델
DecisionTree = DecisionTreeRegressor(random_state=0)
RandomForest = RandomForestRegressor(random_state=0)
XGBoost = xgboost.XGBRegressor(random_state=0)
LightGBM = lightgbm.LGBMRegressor(random_state=0)
use_models = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
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
# print(f"원본 데이터 프레임 shape: {orgdf.shape}")



#################################################대시보드 구성

colors = {
    'background': '#000000',
    'text': '#D2691E',
    "divbackground": '#323232',
    "checkbackground": '#464646'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DjangoDash('dist', external_stylesheets=external_stylesheets)
app.layout = html.Div([
    ####################### 1단
    html.Div([
        ###변수분포
        html.Div([
            #제목
            html.H4(
                "인적, 검사, 처방 정보를 활용한 변수 분포 확인",
            ),

            #드롭다운
            html.Div([
                html.Div([
                    html.H5("모든 변수"),
                    dcc.Dropdown(
                        orgcols,
                        "나이",
                        id="xaxis_histogram",
                    ),
                ], style={
                    "backgroundColor":colors['checkbackground'],
                    "padding":10,
                    "margin":"0px 5px 0px 0px",
                    "flex":1,                    
                }),
                html.Div([
                    html.H5("범주형 변수"),
                    dcc.Dropdown(
                        ["성별", "ica", "ketone(urine)", 'a-glucosidaseinhibitor', 'dppiv','meglitinide', 'metformin', 'sglt2i', 'su', 'tzd', '인슐린외처방내역(glp1-ra)', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)', '지속형+GLP1-RA사용여부'],
                    "성별",
                    id="histogram_color",
                    ),
                ], style={
                    "backgroundColor":colors['checkbackground'],
                    "padding":10,
                    "margin":"0px 0px 0px 5px",
                    "flex":1,
                }),
            ], style={
                "display":"flex",
                "flex-direction":"row"
            }),

            #분포그래프
            html.Div(
                dcc.Graph(
                    id="histogram",
                    style={
                        "margin":"10px 0px 0px 0px",
                    }
                )
            ),
        ], style={
            "backgroundColor":colors['divbackground'],
            "color":colors['text'],            
            # "flex":1,
            "margin":"5px 5px 5px 10px",
            "padding": "0px 20px 20px 20px"
        }),

        ###상관관계    
        html.Div([
            #제목
            html.Div(
                html.H4("인적, 검사, 처방정보를 활용한 상관관계 확인"),
            ),

            #변수체크박스 + 드롭다운
            html.Div([
                #체크박스
                html.Div([                    
                    html.Div([
                        html.H5("연속형 변수"),
                        dcc.Checklist(
                            checkcols,
                            ["alt", "ast"],
                            labelStyle={"display":"inline-block"},
                            id="pair_checklist",
                        ),
                    ], style={
                        "backgroundColor":colors["checkbackground"],
                        "padding": 10
                    }),
                    html.Div([
                        html.H5("범주형 변수"),
                        dcc.Dropdown(
                            ["성별", "ica", "ketone(urine)", 'a-glucosidaseinhibitor', 'dppiv','meglitinide', 'metformin', 'sglt2i', 'su', 'tzd', '인슐린외처방내역(glp1-ra)', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)', '지속형+GLP1-RA사용여부'],
                            "성별",
                            id="pair_color",
                        ),
                    ],  style={
                        "backgroundColor":colors["checkbackground"],
                        "padding": 10,
                        "margin": "10px 0px 0px 0px"
                    }),
                ], style={
                    "flex":0.5,
                }),

                #그래프
                html.Div(
                    dcc.Graph(id="pairplot", figure={"layout":{"height":500}}),
                    style={
                        "flex":2,
                        "margin":"0px 0px 0px 10px",
                }),
            ], style={
                "display": "flex",
                "flex-direction":"row",
            }),
        ], style={
            "backgroundColor":colors['divbackground'],
            "color":colors['text'],            
            # "flex":2,
            "margin": "5px 0px 5px 5px",
            "padding":"0px 20px 0px 20px",
        })


    ], style={
        # "display":"flex",
        # "flex-direction":"row",
        "margin": "10px 20px 5px 5px",
    }),
])


#변수분포
#개별변수 분포
@app.callback(
    Output("histogram", "figure"),
    Input("xaxis_histogram", "value"),
    Input("histogram_color", "value")
    )
def get_histogram(xaxis_histogram, histogram_color):
    fig = px.histogram(orgdf, x=xaxis_histogram, color=histogram_color, marginal="box")
    fig.update_layout(
        margin_t=20, margin_r=20, margin_b=20, margin_l=20,
        plot_bgcolor=colors['divbackground'],paper_bgcolor=colors['divbackground'],font_color=colors['text'],
    )
    return fig

#상관관계
@app.callback(
    Output("pairplot", "figure"),
    Input("pair_checklist", "value"),
    Input("pair_color", "value"))
def get_pairplot(pair_checklist, pair_color):
    fig = px.scatter_matrix(orgdf, dimensions=pair_checklist, color=pair_color)
    fig.update_layout(
        margin_t=20, margin_r=20, margin_b=20, margin_l=20,
        plot_bgcolor=colors['divbackground'],paper_bgcolor=colors['divbackground'], font_color=colors['text'],)
    return fig


# if __name__ == "__main__":
#     app.run_server(debug=True)