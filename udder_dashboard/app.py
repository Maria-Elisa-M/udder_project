# imports
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback,  dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import json


# data sources
pc_path = r"D:\data_store\poject_data\udder_project_gpu\point_clouds"
feature_path = r"D:\data_store\poject_data\udder_project_gpu\features_dict"
udder_pc_path = os.path.join(pc_path, "udder")
quarter_pc_path = os.path.join(pc_path, "quarters")
keypoint_pc_path = os.path.join(pc_path, "keypoints")
teat_pc_path = os.path.join(pc_path, "teat")
teat_len_path = os.path.join(feature_path,  "teat_length")
distance_path = os.path.join(feature_path,  "distance")
filenames = [file.replace(".json", "") for file in os.listdir(teat_len_path)]

color_dict = {'lf': 'cyan', 'rf': 'skyblue', 'lb': 'royalblue', 'rb': 'dodgerblue', 'front':'cyan', 
             'right': 'skyblue', 'left': 'royalblue', 'back': 'dodgerblue', 'udder': 'plum'}

file_dict = {}
for file in filenames:
    cow = file.split("_")[0]
    frame = file.split("_")[-1]
    if cow in set(file_dict.keys()):
        file_dict[cow][frame] = file
    else:
        file_dict[cow] ={frame: file}

cow_list = np.unique(list(file_dict.keys()))
df = pd.read_csv(os.path.join("data", "feature_table.csv"))


keyword_dict = [{'label':'volume', 'value':'vol'},
                {'label':'surface area', 'value': 'sarea'},
                {'label':'circularity', 'value':'circ'},
                {'label':'excentricity', 'value': 'exc'},
                {'label': 'Euclidean distance', 'value': 'eu'},
                {'label': 'geodesic distance', 'value': 'gd'},
                {'label': 'teat length', 'value': 'len'}]
statvar_dict = [{'label': 'mean', 'value': 'mean'},{'label': 'median', 'value': 'median'},]


def list_columns(df, keyword):
    column_list = [col for col in df.columns if keyword in col]
    return column_list

def melt_frame(df, cols):
    selected_df = df[cols]
    melted_df = pd.melt(df, id_vars=['cow', 'frame'], value_vars=cols)
    return melted_df

def subset_df(df, keyword):
    kkword = '_' + keyword
    col = list_columns(df, keyword)
    melted_df = melt_frame(df, cols)
    melted_df["variable"] = [val.replace(kkword, "") for val in melted_df["variable"]]
    return melted_df

def box_fig(df, keyword, statvar):
    kkword = '_' + keyword
    cols = list_columns(df, kkword)
    melted_df = melt_frame(df, cols)
    melted_df["variable"] = [val.replace(kkword, "") for val in melted_df["variable"]]
    melted_df2 = melted_df[~melted_df.variable.isna()]
    group_df = melted_df.drop(["frame"], axis = 1).groupby(["cow", "variable"]).agg(["mean", "median"]).reset_index()
    group_df.columns = [c[1] if len(c[1])>0 else c[0] for c in group_df.columns]
    
    variable_list = np.unique(group_df.variable)
    fig = go.Figure()
    for variable in variable_list:
        dff = group_df[group_df.variable == variable]
        fig.add_trace(go.Box(x = dff[statvar], marker_color = color_dict[variable], name = variable, hovertext = dff['cow']))
    fig.update_layout(paper_bgcolor="black", font_color = "white", plot_bgcolor = "black")
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    return fig

def box_cow_fig(df, keyword, cow):
    kkword = '_' + keyword
    cow_num = int(cow)
    cols = list_columns(df, kkword)
    melted_df = melt_frame(df, cols)
    melted_df["variable"] = [val.replace(kkword, "") for val in melted_df["variable"]]
    melted_df2 = melted_df[~melted_df.variable.isna()]
    cow_df = melted_df2[melted_df2.cow == cow_num]
    
    variable_list = np.unique(cow_df.variable)
    fig = go.Figure()
    for variable in variable_list:
        dff = cow_df[cow_df.variable == variable]
        fig.add_trace(go.Box(x = dff["value"], marker_color = color_dict[variable], name = variable, hovertext = dff['cow']))
    fig.update_layout(paper_bgcolor="black", font_color = "white", plot_bgcolor = "black")
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    return fig


def udder_plot(file):
    udder_pc = np.load(os.path.join(udder_pc_path, file +".npy"))

    with open(os.path.join(quarter_pc_path, file + ".json")) as f:
        quarter_dict = json.load(f)
    
    with open(os.path.join(keypoint_pc_path, file + ".json")) as f:
        keypoint_dict = json.load(f)
    
    with open(os.path.join(distance_path, file + ".json")) as f:
        distance_dict = json.load(f)
    
    with open(os.path.join(teat_pc_path, file + ".json")) as f:
        teat_pc_dict = json.load(f)
    
    with open(os.path.join(teat_len_path, file + ".json")) as f:
        teat_len_dict = json.load(f)
    
    kploc = np.zeros((5,3)) # use five so the Euclidean distance lines close
    for i, teat in enumerate(["lf", "rf", "rb", "lb", "lf"]):
        kploc[i, :] = keypoint_dict[teat]["xyz_tf"]
    
    bottoms = np.zeros((4,3))
    tips = np.zeros((4,3))
    lines = {}
    for i, key in enumerate(teat_len_dict.keys()):
        bottoms[i, :] = teat_len_dict[key]["bottom"]
        tips[i, :] = teat_len_dict[key]["tip"]
        lines[key] = np.row_stack([teat_len_dict[key]["tip"], teat_len_dict[key]["bottom"]])
    
    points = udder_pc
    fig =  go.Figure(data=[go.Scatter3d(x = points[:, 0], y = points[:, 1], z=points[:, 2],mode='markers',
     marker=dict(size=2, color=points[:, 2], colorscale='Viridis', opacity=0.8), name = "Udder")])
    
    for i, key in enumerate(quarter_dict):
        sl = True if i == 0 else False
        points = np.array(quarter_dict[key])
        c = color_dict[key]
        fig.add_trace(go.Scatter3d(x= points[:, 0], y = points[:, 1], z=points[:, 2], mode='markers', marker=dict(color=c, size = 2), name = "quarters", legendgroup = "quarters", showlegend = sl))
    
    for i, key in enumerate(distance_dict):
        sl = True if i == 0 else False
        points = np.array(distance_dict[key]['path'])
        fig.add_trace(go.Scatter3d(x= points[:, 0], y = points[:, 1], z=points[:, 2], mode='markers', marker=dict(color='red', size = 2), name = "geodesic", legendgroup= "geodesic", showlegend = sl))
    
    for i, key in enumerate(teat_pc_dict.keys()):
        sl = True if i == 0 else False
        name = 'obs_pts'
        points = np.array(teat_pc_dict[key][name])
        fig.add_trace(go.Scatter3d(x= points[:, 0], y = points[:, 1], z=points[:, 2], mode='markers', marker=dict(color='white', size = 2), name = name, legendgroup= name, showlegend = sl))
        name = 'pred_pts'
        points = np.array(teat_pc_dict[key][name])
        fig.add_trace(go.Scatter3d(x= points[:, 0], y = points[:, 1], z=points[:, 2], mode='markers', marker=dict(color='white', size = 2), name = name, legendgroup= name, showlegend = sl))
        
    for i, key in enumerate(lines.keys()):
        sl = True if i == 0 else False
        data = lines[key]
        fig.add_trace(go.Scatter3d(x = data[:, 0], y = data[:, 1], z= data[:, 2], mode='lines', line=dict(color="red"), name = "teat_len", showlegend = sl))
    
    fig.add_trace(go.Scatter3d(x = tips[:, 0], y = tips[:, 1], z= tips[:, 2], mode='markers', marker=dict(color="red", size = 4), name = "teat_point", legendgroup ="teat_point", showlegend = False))
    fig.add_trace(go.Scatter3d(x = bottoms[:, 0], y = bottoms[:, 1], z= bottoms[:, 2], mode='markers', marker=dict(color="red", size = 4), name = "teat_point", legendgroup ="teat_point"))
        
    points = kploc
    fig.add_trace(go.Scatter3d(x= points[:, 0], y = points[:, 1], z=points[:, 2], mode='lines', line=dict(color="white", width = 2), name = "euclidean"))
    
    fig.update_layout(paper_bgcolor="black", font_color = "white", plot_bgcolor = "black")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    fig.update_layout(legend_font_color="white", width=1500, height=1000)
    
    return fig 

def blank_fig():
    fig = go.Figure(go.Scatter3d(x=[], y = [], z=[]))
    fig.update_layout(paper_bgcolor="black")
    fig.update_layout(legend_font_color="white")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    fig.update_layout(legend_font_color="white", width=1500, height=1000)
    return fig


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

MENU_STYLE = {
    'backgroundColor': 'black',
    'color': 'white',
}

sidebar = html.Div(
    [
        html.H2("Udder", className="display-4"),
        html.Hr(),
        html.P(
            "choose a cow, months pregnant, and video frame", className="lead"
        ),
        html.Label("Feature:"),
        dcc.Dropdown(id='kw-dpdn',options= keyword_dict, value = 'vol', style = MENU_STYLE),
        dcc.RadioItems(id = 'stat-button', options=statvar_dict, value='median'),
        
        html.Label("Cow ID:"),
        dcc.Dropdown(id='cows-dpdn',options= cow_list, value = '1003', style = MENU_STYLE),
        
        html.Label("Frame:"),
        dcc.Dropdown(id='frame-dpdn', options=[], style = MENU_STYLE),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(
[html.Div(
             [dbc.Row(
                [dbc.Col([dcc.Graph(id='graph', figure = blank_fig())], md = 12),]),
              ])
], id="page-content", style=CONTENT_STYLE)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(
    Output('frame-dpdn', 'options'),
    Input('cows-dpdn', 'value'))
def get_frames(cow):
    global file_dict
    frame_list = list(file_dict[cow].keys())
    return [{'label': c, 'value': c} for c in frame_list]


@app.callback(
    Output("graph", "figure"), 
    Input('cows-dpdn', 'value'),
    Input('frame-dpdn', 'value'))
def update_bar_chart(cow, frame):
    global file_dict
    filename = file_dict[cow][frame]
    fig = udder_plot(filename)
    return fig

@app.callback(
    Output("graph2", "figure"), 
    Input('kw-dpdn', 'value'), 
    Input('stat-button', 'value'))
def update_bar_chart(keyword, statvar):
    global df
    fig = box_fig(df, keyword, statvar)
    return fig

@app.callback(
    Output("graph3", "figure"), 
    Input('kw-dpdn', 'value'), 
    Input('cows-dpdn', 'value'))
def update_bar_chart(keyword, statvar):
    global df
    fig = box_cow_fig(df, keyword, cow)
    return fig

if __name__ == '__main__':
    app.run()