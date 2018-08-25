import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from dash.dependencies import Input, Output
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = dash.Dash(__name__)
app.layout = html.Div(
    html.Div([
        html.H4('DinoAI'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-graph',
            style={
            'height': '200px'
        }),
        dcc.Interval(
            id='interval-component',
            interval=0.1 * 1000,  # in milliseconds
            n_intervals=0
        ),
        html.Div(
            id='live-generations',
            style={
                'float': 'left',
                'width': '80%'
            }
        ),
        html.Div(
            id='live-network',
            style={
                'float': 'right',
                'width': '20%'
            }
        ),
        html.Div(
            id='live-logs',
            style={
                'width': '100%'
            }
        )
    ])
)


@app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    global score, size, speed, position, i_gen, i_generation,  net_value, mode, net_mode, folder, n_genes, \
        n_generation, ypos, logs
    F = open("./tmp/dash_data.txt", "r")
    datos = F.readlines()
    F.close()
    score = datos[0]
    size = datos[1]
    speed = datos[2]
    position = datos[3]
    i_gen = int(datos[4]) + 1
    i_generation = datos[5]
    net_value = datos[6]
    mode = datos[7]
    net_mode = datos[8]
    folder = datos[9]
    n_genes = datos[10]
    n_generation = datos[11]
    ypos = datos[12]
    logs = datos[13]

    style = {'padding': '5px', 'fontSize': '16px',  'font-family': 'Arial, Helvetica, sans-serif'}
    return [
        html.P(f'Score: {score}', style=style),
        html.P(f'Size: {size}', style=style),
        html.P(f'Speed: {speed}', style=style),
        html.P(f'X position: {position}', style=style),
        html.P(f'Y position: {ypos}',  style = style)

    ]

@app.callback(Output('live-generations', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_generations(n):
    global net_value, mode, net_mode, folder, logs

    log_lines = logs.split("<br/>")

    style = {'padding': '5px', 'fontSize': '16px',  'font-family': 'Arial, Helvetica, sans-serif'}
    return [
        html.P(f'Mode: {mode}', style=style),
        html.P(f'Net value: {net_value}', style=style),
        html.P(f'Net mode: {net_mode}', style=style),
        html.P(f'Folder: {folder}', style=style)
    ]

@app.callback(Output('live-network', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_network(n):
    global score, size, speed, position, i_gen, i_generation, n_generation, n_genes

    style = {'padding': '5px', 'fontSize': '16px',  'font-family': 'Arial, Helvetica, sans-serif'}
    return [
        html.P(f'Generation: {i_generation}/{n_generation}', style=style),
        html.P(f'Gen: {i_gen}/{n_genes}', style=style),
    ]

@app.callback(Output('live-logs', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_logs(n):
    global logs
    style = {'padding': '5px', 'fontSize': '16px', 'font-family': 'Arial, Helvetica, sans-serif', 'width': '100%'}
    return [
        html.Iframe(srcDoc=logs, style=style)
    ]


@app.callback(Output('live-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    global score, size, speed, position
    score = float(score)
    speed = float(speed)
    position = float(position)

    position_plot = position/500
    score_plot = score / 10000
    speed_plot = speed / 20
    data = [go.Bar(
        x=['Score', 'Speed', 'Position'],
        y=[score_plot, speed_plot, position_plot],
        x0=[0, 0, 0],
    )]

    layout = go.Layout(
        yaxis=dict(
            range=[0, 1],
            ticks="",
            visible=False
        ),

    )

    fig = go.Figure(data = data, layout=layout)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
