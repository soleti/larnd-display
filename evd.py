#!/usr/bin/env python3

"""
Web-based event display for ArgonCube detectors
"""

import shutil
import atexit

from pathlib import Path

import fire
import h5py
import numpy as np

import dash
import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
from plotly import subplots

from larndsim.consts import detector
from larnd_display.display_utils import DetectorGeometry, plot_geometry, plot_hits, plot_light, plot_tracks

EVENT_DIVIDERS = None
MC_PACKETS = None
PACKETS = None
LIGHT_LUT = None
MY_GEOMETRY = None
TRACKS = None
UPLOAD_FOLDER_ROOT = "cache"

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
                external_scripts=[
                    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML",
                ],
                title="LArPix event display")

def draw_event(event_id):
    start_packet = EVENT_DIVIDERS[event_id]
    end_packet = EVENT_DIVIDERS[event_id+1]

    track_ids = np.unique(MC_PACKETS[start_packet:end_packet]["track_ids"])[1:]
    last_trigger = PACKETS[start_packet]["timestamp"]
    event_packets = PACKETS[start_packet:end_packet]
    drawn_objects = plot_hits(MY_GEOMETRY, event_packets, start_packet, last_trigger)
    drawn_objects.extend(plot_tracks(TRACKS, range(track_ids[0], track_ids[-1]), event_id))
    drawn_objects.extend(plot_geometry())

    if LIGHT_LUT:
        drawn_objects.extend(plot_light(MY_GEOMETRY,np.sum(LIGHT_LUT[track_ids]["n_photons_det"],axis=0)))

    return drawn_objects

# @app.callback(Input("event-display", "clickData"),
#               State("event-display", "figure"))
# def test(click_data, figure):
#     if click_data:
#         print(click_data)

@app.callback(
    [Output("event-display", "figure"),
     Output("alert-auto", "is_open"),
     Output("input-evid", "value"),],
    Input("input-evid", "value"),
    State("event-display", "figure"))
def update_output(event_id, figure):
    fig = go.Figure(figure)

    if EVENT_DIVIDERS is None:
        return fig, False, 0

    show_alert = False
    event_id = int(event_id)

    if event_id >= len(EVENT_DIVIDERS)-1:
        event_id = len(EVENT_DIVIDERS)-2
        show_alert = True
        return fig, True, event_id

    if event_id < 0:
        event_id = 0
        show_alert = True
        return fig, True, event_id

    # if click_data:
    #     print(click_data)

    fig.data = []
    fig.add_traces(draw_event(event_id))

    return fig, show_alert, event_id

@app.callback(
    [Output("object-information", "children"),
     Output("time-histogram", "figure"),
     Output("time-histogram", "style")],
    Input("input-evid", "value"))
def histogram(n_events):
    if EVENT_DIVIDERS is not None:
        start_packet = EVENT_DIVIDERS[n_events]
        end_packet = EVENT_DIVIDERS[n_events+1]
        event_packets = PACKETS[start_packet:end_packet]
        event_packets = event_packets[event_packets["packet_type"]==0]
        anodes = []

        for io_group,io_channel in zip(event_packets["io_group"],event_packets["io_channel"]):
            if not io_group % 4:
                this_io_group = 4
            else:
                this_io_group = io_group % 4
            tile_id = MY_GEOMETRY.get_tile_id(this_io_group,io_channel)
            if tile_id in detector.TILE_MAP[0][0] or tile_id in detector.TILE_MAP[0][1]:
                anodes.append(0)
            else:
                anodes.append(1)

        anodes = np.array(anodes)

        n_modules = len(detector.MODULE_TO_IO_GROUPS.keys())
        start_t = PACKETS[start_packet]["timestamp"]
        active_modules = []

        for module_id in range(n_modules):
            query = ((event_packets["io_group"]-1)//4==module_id)
            if len(event_packets[query]) == 0:
                continue

            active_modules.append(module_id)

        histos = subplots.make_subplots(rows=len(active_modules), cols=2,
                                        subplot_titles=["(%i,%i)" % (m+1,p+1) for m in active_modules for p in range(2)],
                                        vertical_spacing=0.25/len(active_modules) if len(active_modules) else 0,
                                        shared_xaxes=True, shared_yaxes=True)

        for im, module_id in enumerate(active_modules):
            query = ((event_packets["io_group"]-1)//4==module_id)

            histo1 = go.Histogram(
                x=event_packets["timestamp"][(anodes==0) & query]-start_t,
                xbins=dict(
                    start=0,
                    end=3200,
                    size=20
                ),
            )
            histo2 = go.Histogram(
                x=event_packets["timestamp"][(anodes==1) & query]-start_t,
                xbins=dict(
                    start=0,
                    end=3200,
                    size=20
                ),
            )

            histos.append_trace(histo1, im+1, 1)
            histos.append_trace(histo2, im+1, 2)

        histos.update_annotations(font_size=12)
        histos.update_layout(margin=dict(l=0, r=0, t=30, b=10),showlegend=False,template="plotly_white")
        histos.update_xaxes(title_text="Time [timestamp]", row=len(active_modules))
        histos.update_xaxes(linecolor="lightgray",matches="x",mirror=True,ticks="outside",showline=True)
        histos.update_yaxes(linecolor="lightgray",matches="y",mirror=True,ticks="outside",showline=True)
        subplots_height =  "%fvh" % (len(active_modules)*22+5)
        return "", histos, dict(height=subplots_height, display="block")

    return "", go.Figure(), dict(display="none")

@app.callback(Output("total-events", "children"),
              [Input("select-file", "isCompleted"),
               State("select-file", "fileNames"),
               State("select-file", "upload_id")])
def upload_file(is_completed, filenames, upload_id):
    global EVENT_DIVIDERS
    global MC_PACKETS
    global PACKETS
    global LIGHT_LUT
    global MY_GEOMETRY
    global TRACKS

    if not is_completed:
        return None

    if filenames is not None:
        if upload_id:
            root_folder = Path(UPLOAD_FOLDER_ROOT) / upload_id
        else:
            root_folder = Path(UPLOAD_FOLDER_ROOT)

        h5_file = root_folder / filenames[0]
        datalog = h5py.File(h5_file, "r")
        TRACKS = datalog["tracks"]
        PACKETS = datalog["packets"]
        MC_PACKETS = datalog["mc_packets_assn"]

        if "light_dat" in datalog.keys():
            LIGHT_LUT = datalog["light_dat"]

        trigger_packets = np.argwhere(PACKETS["packet_type"] == 7).T[0]
        EVENT_DIVIDERS = trigger_packets[:-1][np.diff(trigger_packets) != 1]
        EVENT_DIVIDERS = np.append(EVENT_DIVIDERS, [trigger_packets[-1], len(PACKETS)])

        return f"/ {len(EVENT_DIVIDERS)-2}"

    return None

def run_display(detector_properties, pixel_layout):
    global MY_GEOMETRY

    MY_GEOMETRY = DetectorGeometry(detector_properties, pixel_layout)

    fig = go.Figure(plot_geometry())
    camera = dict(
        eye=dict(x=-2,y=0.3,z=1.1)
    )

    fig.update_layout(scene_camera=camera,
                      uirevision=True,
                      margin=dict(l=0, r=0, t=4),
                      legend={"y" : 0.8},
                      scene=dict(
                          xaxis=dict(backgroundcolor="white",
                                     showspikes=False,
                                     showgrid=False,
                                     title="x [mm]"),
                          yaxis=dict(backgroundcolor="white",
                                     showgrid=False,
                                     showspikes=False,
                                     title="z [mm]"),
                          zaxis=dict(backgroundcolor="white",
                                     showgrid=False,
                                     showspikes=False,
                                     title="y [mm]")))

    app.layout = dbc.Container(
        fluid=True,
        style={"padding": "1.5em"},
        children=[
            dbc.Row([
                dbc.Col([
                    html.H1(children="LArPix event display"),
                    du.Upload(id="select-file",
                                text="Drag and drop or click here to upload",
                                max_file_size=10000,
                                chunk_size=50,
                                default_style={"width": "15em", "padding": "0", "margin": "0"},
                                pause_button=True,
                                filetypes=["h5"]),
                ], width=9),
                dbc.Col([html.Img(src="https://github.com/soleti/larnd-display/raw/main/docs/logo.png",
                                  style={"height":"8em"})], width=3, style={"text-align":"right"}),
            ]),
            dbc.Row([
                html.Div([
                    html.P(children="Event: ", style={"display":"inline-block", "margin-top":"1em"}),
                    dcc.Input(id="input-evid",
                              type="number",
                              placeholder="0",
                              value="0",
                              style={"width": "5em",
                                     "display": "inline-block",
                                     "margin-right": "0.5em",
                                     "margin-left": "0.5em"}),
                    html.Div(id="total-events",
                             style={"padding-right": "1em",
                                    "display":"inline-block",
                                    "text-align":"center"}),
                ]),
            ],style={"margin-left": "0.2em","margin-bottom":"1em"}),
            dbc.Row([
                dbc.Alert(
                        "You have reached the end of the file",
                        id="alert-auto",
                        is_open=False,
                        color="warning"
                ),
            ], style={"padding-left": "0.2em"}),
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(id="event-display",
                                       figure=fig,
                                       clear_on_unhover=True,
                                       style={"height": "75vh"})],
                            width=7),
                    dbc.Col([html.Div([html.P(id="object-information"),
                                       dcc.Graph(id="time-histogram", style={"display": "none"})])],
                            width=5),
                ]
            )
        ]
    )

    app.run_server(port=8000, host="127.0.0.1")

    return app

@atexit.register
def clean_cache():
    try:
        print("Cleaning cache...")
        shutil.rmtree(UPLOAD_FOLDER_ROOT)
    except OSError as err:
        print("Error: %s : %s" % (UPLOAD_FOLDER_ROOT, err.strerror))

if __name__ == "__main__":
    du.configure_upload(app, UPLOAD_FOLDER_ROOT)
    fire.Fire(run_display)
