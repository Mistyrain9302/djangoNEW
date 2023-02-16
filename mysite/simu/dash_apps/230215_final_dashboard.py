import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fontPath = r"C:\Windows\Fonts\NIAGENG.TTF"
fontName = fm.FontProperties(fname=fontPath, size=12).get_name()
plt.rc("font", family=fontName)
mpl.rcParams["axes.unicode_minus"] = False

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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_excel(r"C:\Users\ASIA-03\Desktop\Mypj\final_pivot_df48.xlsx")
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
# print(usesc)

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
# print(f"원본 데이터 프레임 shape: {orgdf.shape}")



#################################################대시보드 구성

colors = {
    'background': '#d2d2d2',
    'text': '#000000'
}


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    ####################### 제목
    html.H1(
        "제 2형 당뇨 환자 데이터를 활용한 합병증 예측",
        style={'textAlign': 'center',
               'backgroundColor': colors['background'],
               'color': colors["text"]
        }),

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



    ########################### 2단
    html.Div([
        ###군집화
        html.Div([
            html.H4("군집분석"),
            #인적정보선택
            html.Br(),
            html.Div([
                html.H5("인적정보"),
                dcc.Checklist(
                    humancols,
                    ["나이", "몸무게", "성별", "bmi"],
                    id = "cluster_human_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),
            #검사정보선택
            html.Br(),
            html.Div([
                html.H5("검사정보"),
                dcc.Checklist(
                    checkcols,
                    [],
                    id = "cluster_check_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),
            #약처방정보선택
            html.Br(),
            html.Div([
                html.H5("약처방정보"),
                dcc.Checklist(
                    medcols,
                    ['a-glucosidaseinhibitor', 'dppiv', 'meglitinide', 'metformin', 'sglt2i', 'su', 'tzd', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)'],
                    id = "cluster_med_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),
            #클러스터 개수 설정
            html.Br(),
            html.Div([
                html.H5("군집 수"),
                dcc.Input(id="input_k", type="text", value=1)
            ]),
            #실행버튼
            html.Br(),
            html.Button("실행", id="submit_button1", n_clicks=0)
        ], style={"width":"20%", "padding":"20px"}),

        ###군집분석 그래프
        html.Div(
            dcc.Graph(id="cluster_scatter_plot", figure={"layout":{"width":600, "height":600}}
            )
        ),

        ###Feature Importance
        html.Div([
            html.H4("군집내 영향도"),
            #인적정보선택
            html.Br(),
            html.Div([
                html.H5("인적정보"),
                dcc.Checklist(
                    humancols,
                    ["나이", "몸무게", "성별", "bmi"],
                    id = "clf_human_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),
            #검사정보선택
            html.Br(),
            html.Div([
                html.H5("검사정보"),
                dcc.Checklist(
                    checkcols,
                    checkcols,
                    id = "clf_check_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),
            #약처방정보선택
            html.Br(),
            html.Div([
                html.H5("약처방정보"),
                dcc.Checklist(
                    medcols,
                    [],
                    id = "clf_med_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),
            #클러스터 개수 설정
            html.Br(),
            html.Div([
                html.H5("군집 수"),
                dcc.Input(id="input_k2", type="text", value=2)
            ]),
            
            #실행버튼
            html.Br(),
            html.Button("실행", id="submit_button2", n_clicks=0)
        ], style={"width":"20%", "padding":"20px"}),

        ###featureimportance 그래프
        html.Div(
            dcc.Graph(id="clf_scatter_plot", figure={"layout":{"width":600, "height":600}}
            )
        )
    ], style={"display":"flex", "flex-direction":"row"}),



    ########################### 3단
    html.Div([
        ###회귀분석
        html.Div([
            html.H4("회귀분석"),
            #인적정보선택
            html.Br(),
            html.Div([
                html.H5("인적정보"),
                dcc.Checklist(
                    humancols,
                    ["나이", "몸무게", "성별", "bmi"],
                    id = "reg_human_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),

            #검사정보선택
            html.Br(),
            html.Div([
                html.H5("검사정보"),
                dcc.Checklist(
                    checkcols,
                    checkcols,
                    id = "reg_check_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),

            #약처방정보선택
            html.Br(),
            html.Div([
                html.H5("약처방정보"),
                dcc.Checklist(
                    medcols,
                    ["tzd"],
                    id = "reg_med_check",
                    labelStyle={"display":"inline-block"}
                )
            ]),
            
            #예측할 변수 선택
            html.Br(),
            html.Div([
                html.H5("예측할 변수"),
                dcc.RadioItems(
                    checkcols,
                    "cr",
                    id = "reg_target_radio",
                    labelStyle={"display":"inline-block"}
                )
            ]),

            #모델선택
            html.Br(),
            html.Div([
                html.H5("사용모델"),
                dcc.Checklist(
                    ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"],
                    ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"],
                    id="use_models"
                )
            ]),

            

        ], style={"width":"20%", "padding":"20px"}),


        ###예측값 그래프
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id="reg_plot1", figure={"layout":{"width":600, "height":400}}
                    ),
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("r2"),
                                html.Th("mae"),
                                html.Th("mse"),
                                html.Th("rmse"),
                            ])
                        ),
                        
                        html.Tbody(
                            html.Tr([
                                html.Td(id="reg1_r2"),
                                html.Td(id="reg1_mae"),
                                html.Td(id="reg1_mse"),
                                html.Td(id="reg1_rmse"),
                            ])
                        )
                    ],  style={'marginLeft': 'auto', 'marginRight': 'auto', "width": 300})
                ]),
                
                html.Div([
                    dcc.Graph(id="reg_plot2", figure={"layout":{"width":600, "height":400}}
                    ),
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("r2"),
                                html.Th("mae"),
                                html.Th("mse"),
                                html.Th("rmse"),
                            ])
                        ),
                        
                        html.Tbody(
                            html.Tr([
                                html.Td(id="reg2_r2"),
                                html.Td(id="reg2_mae"),
                                html.Td(id="reg2_mse"),
                                html.Td(id="reg2_rmse"),
                            ])
                        )
                    ],  style={'marginLeft': 'auto', 'marginRight': 'auto', "width": 300})
                ]),
            ], style={"display":"flex", "flex-direction":"row"}),
            
            
            html.Div([
                html.Div([
                    dcc.Graph(id="reg_plot3", figure={"layout":{"width":600, "height":400}}
                    ),
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("r2"),
                                html.Th("mae"),
                                html.Th("mse"),
                                html.Th("rmse"),
                            ])
                        ),
                        
                        html.Tbody(
                            html.Tr([
                                html.Td(id="reg3_r2"),
                                html.Td(id="reg3_mae"),
                                html.Td(id="reg3_mse"),
                                html.Td(id="reg3_rmse"),
                            ])
                        )
                    ],  style={'marginLeft': 'auto', 'marginRight': 'auto', "width": 300})
                ]),
                html.Div([
                    dcc.Graph(id="reg_plot4", figure={"layout":{"width":600, "height":400}}
                    ),
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("r2"),
                                html.Th("mae"),
                                html.Th("mse"),
                                html.Th("rmse"),
                            ])
                        ),
                        
                        html.Tbody(
                            html.Tr([
                                html.Td(id="reg4_r2"),
                                html.Td(id="reg4_mae"),
                                html.Td(id="reg4_mse"),
                                html.Td(id="reg4_rmse"),
                            ])
                        )
                    ],  style={'marginLeft': 'auto', 'marginRight': 'auto', "width": 300})
                ])    
            ], style={"display":"flex", "flex-direction":"row"})
        ])
    ], style={"display":"flex", "flex-direction":"row"})    
])

#################################################### 별도함수
#클러스터 나누기
def get_cluster(df, k):
    km = KMeans(n_clusters=int(k), random_state=0)
    km.fit(df)
    labels = km.labels_
    cluster_col_name = f"cluster{k}"
    df[cluster_col_name] = labels
    return df, cluster_col_name

#분류데이터분리
def get_train_test_data_clf(df, dropcols):
    from sklearn.model_selection import train_test_split
    X = df.drop(dropcols, axis=1)
    y = df[dropcols]    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0, stratify=y)
    return x_train, x_test, y_train, y_test

#회귀데이터분리
def get_train_test_data_reg(df, dropcols):
    from sklearn.model_selection import train_test_split
    X = df.drop(dropcols, axis=1)
    y = df[dropcols]    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
    return x_train, x_test, y_train, y_test

#회귀함수
def get_compare_plot(df, model): 
    fig = go.Figure()
    for col in list(df):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col))
        fig.update_layout(title=dict(text=model.__class__.__name__))
    return fig

#################################################### 콜백

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

#군집분석
@app.callback(
    Output("cluster_scatter_plot", "figure"),    
    Input("submit_button1", "n_clicks"),
    Input("cluster_human_check", "value"),
    Input("cluster_check_check", "value"),
    Input("cluster_med_check", "value"),
    State("input_k", "value"),    
    )    
def get_cluster_plot(submit, cluster_human_check, cluster_check_check, cluster_med_check, k):
    fig = go.Figure()    
    
    totalcols = cluster_human_check + cluster_check_check + cluster_med_check #군집에 사용할 컬럼들        
    dff = data[totalcols]        
    dff = dff.dropna()

    km = KMeans(n_clusters=int(k), random_state=0)
    km.fit(dff)
    labels = km.labels_
    cluster_col_name = f"cluster{k}"
    dff[cluster_col_name] = labels

    tsne = TSNE(n_components=2, random_state=0)
    arr = tsne.fit_transform(dff.drop(cluster_col_name, axis=1))
    
    for i in range(int(k)):
        fig.add_trace(go.Scatter(x=arr[dff[cluster_col_name]==i, 0], y=arr[dff[cluster_col_name]==i, 1], mode="markers", name=f"cluster{i}"))        
    return fig

#군집영향도
@app.callback(
    Output("clf_scatter_plot", "figure"),    
    Input("submit_button2", "n_clicks"),
    Input("cluster_human_check", "value"),
    Input("cluster_check_check", "value"),
    Input("cluster_med_check", "value"),    
    Input("clf_human_check", "value"),
    Input("clf_check_check", "value"),
    Input("clf_med_check", "value"),            
    State("input_k2", "value"))

#군집 영향도 feature importance
def get_feature_importance(submit, cluster_human_check, cluster_check_check, cluster_med_check, clf_human_check, clf_check_check, clf_med_check, k):    
    fig = go.Figure()
    totalcols = cluster_human_check + cluster_check_check + cluster_med_check #군집에 사용할 컬럼들        
    dff = data[totalcols] 
    dff = dff.dropna()
    dff, cluster_col_name = get_cluster(dff, k) #군집화 된 데이터 프레임
    # print(dff.shape)
    
    clfcols = clf_human_check + clf_check_check + clf_med_check
    # print(clfcols)
    clfidx = dff.index
    dff2 = data.loc[clfidx, clfcols]
    dff2[cluster_col_name] = dff[cluster_col_name].values
    dff2 = dff2.dropna()
    # print(dff2.shape)
    
    model = xgboost.XGBClassifier(random_state=42)
    x_train, x_test, y_train, y_test = get_train_test_data_clf(dff2, cluster_col_name)#데이터분리
    
    model.fit(x_train, y_train)    
    importances = model.feature_importances_
    sortIdx = importances.argsort()[::1]
    
    fig = px.bar(x=importances[sortIdx], y=x_train.columns[sortIdx])
    return fig

#회귀
@app.callback(
    Output("reg_plot1", "figure"),    
    Output("reg_plot2", "figure"),
    Output("reg_plot3", "figure"),
    Output("reg_plot4", "figure"),
    
    #dt
    Output("reg1_r2", "children"),
    Output("reg1_mae", "children"),
    Output("reg1_mse", "children"),
    Output("reg1_rmse", "children"),
    
    #rf
    Output("reg2_r2", "children"),
    Output("reg2_mae", "children"),
    Output("reg2_mse", "children"),
    Output("reg2_rmse", "children"),
    
    #xgb
    Output("reg3_r2", "children"),
    Output("reg3_mae", "children"),
    Output("reg3_mse", "children"),
    Output("reg3_rmse", "children"),
    
    #lgbm
    Output("reg4_r2", "children"),
    Output("reg4_mae", "children"),
    Output("reg4_mse", "children"),
    Output("reg4_rmse", "children"),
    
    
    Input("reg_human_check", "value"),
    Input("reg_check_check", "value"),
    Input("reg_med_check", "value"),  
      
    Input("reg_target_radio", "value"), #제거
    Input("use_models", "value") #모델선택
)
def get_regression_plot(reg_human_check, reg_check_check, reg_med_check, reg_target_radio, use_models):
    usecols = humancols + medcols + checkcols
    
    dff = data[usecols] #인적 + 검사 + 약 데이터프레임 다시정의
    # print(dff.columns)
    
    regcols = reg_human_check + reg_check_check + reg_med_check
    # print(regcols)
    dff2 = dff[regcols] #체크된 컬럼만 정의
    dff2 = dff2.dropna()
    # print(dff2.shape)
    
    #데이터분리
    x_train, x_test, y_train, y_test = get_train_test_data_reg(dff2, reg_target_radio)
    
    comparedfs = []
    r2s = []
    maes = []
    mses = []
    rmses = []
    
    for model in use_models:
        if model == "DecisionTree":
            models[0].fit(x_train, y_train)
            pred = models[0].predict(x_test)
            comparedf = pd.DataFrame({"y_test": y_test.values, "pred": pred})
            comparedfs.append(comparedf)
            
            r2 = r2_score(pred, y_test).round(3)
            mae = mean_absolute_error(pred, y_test).round(3)
            mse = mean_squared_error(pred, y_test, squared=True).round(3)
            rmse = mean_squared_error(pred, y_test, squared=False).round(3)
            
            r2s.append(r2)
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            
        elif model == "RandomForest":
            models[1].fit(x_train, y_train)
            pred = models[1].predict(x_test)
            comparedf = pd.DataFrame({"y_test": y_test.values, "pred": pred})
            comparedfs.append(comparedf)
            
            r2 = r2_score(pred, y_test).round(3)
            mae = mean_absolute_error(pred, y_test).round(3)
            mse = mean_squared_error(pred, y_test, squared=True).round(3)
            rmse = mean_squared_error(pred, y_test, squared=False).round(3)
            
            r2s.append(r2)
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            
        elif model == "XGBoost":
            models[2].fit(x_train, y_train)
            pred = models[2].predict(x_test)
            comparedf = pd.DataFrame({"y_test": y_test.values, "pred": pred})
            comparedfs.append(comparedf)
            
            r2 = r2_score(pred, y_test).round(3)
            mae = mean_absolute_error(pred, y_test).round(3)
            mse = mean_squared_error(pred, y_test, squared=True).round(3)
            rmse = mean_squared_error(pred, y_test, squared=False).round(3)
            
            r2s.append(r2)
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            
        elif model == "LightGBM":
            models[3].fit(x_train, y_train)
            pred = models[3].predict(x_test)
            comparedf = pd.DataFrame({"y_test": y_test.values, "pred": pred})
            comparedfs.append(comparedf)
            
            r2 = r2_score(pred, y_test).round(3)
            mae = mean_absolute_error(pred, y_test).round(3)
            mse = mean_squared_error(pred, y_test, squared=True).round(3)
            rmse = mean_squared_error(pred, y_test, squared=False).round(3)
            
            r2s.append(r2)
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
     
        
    return [get_compare_plot(comparedfs[0], models[0]),
            get_compare_plot(comparedfs[1], models[1]),
            get_compare_plot(comparedfs[2], models[2]),
            get_compare_plot(comparedfs[3], models[3]),
            r2s[0], maes[0], mses[0], rmses[0],
            r2s[1], maes[1], mses[1], rmses[1],
            r2s[2], maes[2], mses[2], rmses[2],
            r2s[3], maes[3], mses[3], rmses[3]]



    
if __name__ == "__main__":
    app.run_server(debug=True)
    