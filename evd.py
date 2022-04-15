#!/usr/bin/env python3

"""
Web-based event display for ArgonCube detectors
"""

import shutil
import atexit

from os.path import basename
from pathlib import Path

import fire
import h5py
import numpy as np

import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, DashProxy, Input, State, MultiplexerTransform

import plotly.graph_objects as go
from plotly import subplots

from larndsim.consts import detector
from larnd_display.display_utils import (
    DetectorGeometry,
    plot_geometry,
    plot_hits,
    plot_light,
    plot_tracks,
)

MY_GEOMETRY = None
UPLOAD_FOLDER_ROOT = "cache"

app = DashProxy(
    prevent_initial_callbacks=True,
    transforms=[MultiplexerTransform()],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML",
    ],
    title="LArPix event display",
)


def draw_event(filename, event_id):
    with h5py.File(filename, "r") as datalog:
        tracks = datalog["tracks"]
        packets = datalog["packets"]
        mc_packets = datalog["mc_packets_assn"]

        trigger_packets = np.argwhere(packets["packet_type"] == 7).T[0]
        event_dividers = trigger_packets[:-1][np.diff(trigger_packets) != 1]
        event_dividers = np.append(event_dividers, [trigger_packets[-1], len(packets)])

        start_packet = event_dividers[event_id]
        end_packet = event_dividers[event_id + 1]

        track_ids = np.unique(mc_packets[start_packet:end_packet]["track_ids"])[1:]
        last_trigger = packets[start_packet]["timestamp"]
        event_packets = packets[start_packet:end_packet]
        drawn_objects = plot_hits(
            MY_GEOMETRY, event_packets, start_packet, last_trigger
        )
        drawn_objects.extend(
            plot_tracks(tracks, range(track_ids[0], track_ids[-1]), event_id)
        )
        drawn_objects.extend(plot_geometry())

        if "light_dat" in datalog.keys():
            light_lut = datalog["light_dat"]
            drawn_objects.extend(
                plot_light(
                    MY_GEOMETRY, np.sum(light_lut[track_ids]["n_photons_det"], axis=0)
                )
            )

        return drawn_objects


@app.callback(
    [
        Output("event-display", "figure"),
        Output("alert-auto", "is_open"),
        Output("event-id", "data"),
    ],
    Input("input-evid", "value"),
    [
        State("event-dividers", "data"),
        State("event-display", "figure"),
        State("filename", "data"),
    ],
)
def update_output(event_id, event_dividers, figure, filename):
    fig = go.Figure(figure)

    if event_dividers is None:
        return fig, False, 0

    try:
        event_id = int(event_id)
    except KeyError:
        print("Invalid event id")
        return fig, False, event_id

    show_alert = False
    if event_id >= len(event_dividers) - 1:
        event_id = len(event_dividers) - 2
        show_alert = True
        return fig, True, event_id

    if event_id < 0:
        event_id = 0
        show_alert = True
        return fig, True, event_id

    fig.data = []
    fig.add_traces(draw_event(filename, event_id))

    return fig, show_alert, event_id


@app.callback(
    [
        Output("object-information", "children"),
        Output("time-histogram", "figure"),
        Output("time-histogram", "style"),
    ],
    Input("input-evid", "value"),
    [State("event-dividers", "data"), State("filename", "data")],
)
def histogram(event_id, event_dividers, filename):
    if event_dividers is not None:
        start_packet = event_dividers[event_id]
        end_packet = event_dividers[event_id + 1]

        with h5py.File(filename, "r") as datalog:
            packets = datalog["packets"]

            event_packets = packets[start_packet:end_packet]
            event_packets = event_packets[event_packets["packet_type"] == 0]
            anodes = []

            for io_group, io_channel in zip(
                event_packets["io_group"], event_packets["io_channel"]
            ):
                if not io_group % 4:
                    this_io_group = 4
                else:
                    this_io_group = io_group % 4
                tile_id = MY_GEOMETRY.get_tile_id(this_io_group, io_channel)
                if (
                    tile_id in detector.TILE_MAP[0][0]
                    or tile_id in detector.TILE_MAP[0][1]
                ):
                    anodes.append(0)
                else:
                    anodes.append(1)

            anodes = np.array(anodes)

            n_modules = len(detector.MODULE_TO_IO_GROUPS.keys())
            start_t = packets[start_packet]["timestamp"]
            active_modules = []

            for module_id in range(n_modules):
                query = (event_packets["io_group"] - 1) // 4 == module_id
                if len(event_packets[query]) == 0:
                    continue

                active_modules.append(module_id)

            histos = subplots.make_subplots(
                rows=len(active_modules),
                cols=2,
                subplot_titles=[
                    "(%i,%i)" % (m + 1, p + 1) for m in active_modules for p in range(2)
                ],
                vertical_spacing=0.25 / len(active_modules)
                if len(active_modules)
                else 0,
                shared_xaxes=True,
                shared_yaxes=True,
            )

            for im, module_id in enumerate(active_modules):
                query = (event_packets["io_group"] - 1) // 4 == module_id

                histo1 = go.Histogram(
                    x=event_packets["timestamp"][(anodes == 0) & query] - start_t,
                    xbins=dict(start=0, end=3200, size=20),
                )
                histo2 = go.Histogram(
                    x=event_packets["timestamp"][(anodes == 1) & query] - start_t,
                    xbins=dict(start=0, end=3200, size=20),
                )

                histos.append_trace(histo1, im + 1, 1)
                histos.append_trace(histo2, im + 1, 2)

            histos.update_annotations(font_size=12)
            histos.update_layout(
                margin=dict(l=0, r=0, t=30, b=10),
                showlegend=False,
                template="plotly_white",
            )
            histos.update_xaxes(title_text="Time [timestamp]", row=len(active_modules))
            histos.update_xaxes(
                linecolor="lightgray",
                matches="x",
                mirror=True,
                ticks="outside",
                showline=True,
            )
            histos.update_yaxes(
                linecolor="lightgray",
                matches="y",
                mirror=True,
                ticks="outside",
                showline=True,
            )
            subplots_height = "%fvh" % (len(active_modules) * 22 + 5)
            return "", histos, dict(height=subplots_height, display="block")

    return "", go.Figure(), dict(display="none")


@app.callback(
    Output("filename-text", "children"),
    Input("filename", "modified_timestamp"),
    State("filename", "data"),
)
def update_filename(modified_timestamp, filename):
    if modified_timestamp is None:
        raise PreventUpdate

    if filename:
        filename = html.Span(
            children=[
                html.Span("File: ", style={"font-weight": "bold"}),
                html.Span(basename(filename), style={"font-family": "monospace"}),
            ]
        )
    else:
        filename = "No file selected"

    return filename


@app.callback(
    Output("total-events", "children"),
    Input("event-dividers", "modified_timestamp"),
    State("event-dividers", "data"),
)
def update_total_events(modified_timestamp, event_dividers):
    if modified_timestamp is None:
        raise PreventUpdate

    if not event_dividers:
        total_events = 0
    else:
        total_events = len(event_dividers) - 2

    return f"/ {total_events}"


@app.callback(
    Output("input-evid", "value"),
    Input("event-id", "modified_timestamp"),
    State("event-id", "data"),
)
def update_event_id(modified_timestamp, event_id):
    if modified_timestamp is None:
        raise PreventUpdate

    event_id = event_id or 0
    return int(event_id)


@app.callback(
    [
        Output("filename", "data"),
        Output("event-dividers", "data"),
        Output("event-id", "data"),
    ],
    [
        Input("select-file", "isCompleted"),
        State("filename", "data"),
        State("event-dividers", "data"),
        State("select-file", "fileNames"),
        State("select-file", "upload_id"),
    ],
)
def upload_file(is_completed, filename, event_dividers, filenames, upload_id):

    if not is_completed:
        return filename, event_dividers, 0

    if filenames is not None:
        if upload_id:
            root_folder = Path(UPLOAD_FOLDER_ROOT) / upload_id
        else:
            root_folder = Path(UPLOAD_FOLDER_ROOT)

        h5_file = root_folder / filenames[0]
        datalog = h5py.File(h5_file, "r")
        packets = datalog["packets"]

        trigger_packets = np.argwhere(packets["packet_type"] == 7).T[0]
        event_dividers = trigger_packets[:-1][np.diff(trigger_packets) != 1]
        event_dividers = np.append(event_dividers, [trigger_packets[-1], len(packets)])

        return str(h5_file), event_dividers, 0

    return filename, event_dividers, 0


def run_display(detector_properties, pixel_layout):
    global MY_GEOMETRY

    MY_GEOMETRY = DetectorGeometry(detector_properties, pixel_layout)

    fig = go.Figure(plot_geometry())
    camera = dict(eye=dict(x=-2, y=0.3, z=1.1))

    fig.update_layout(
        scene_camera=camera,
        uirevision=True,
        margin=dict(l=0, r=0, t=4),
        legend={"y": 0.8},
        scene=dict(
            xaxis=dict(
                backgroundcolor="white",
                showspikes=False,
                showgrid=False,
                title="x [mm]",
            ),
            yaxis=dict(
                backgroundcolor="white",
                showgrid=False,
                showspikes=False,
                title="z [mm]",
            ),
            zaxis=dict(
                backgroundcolor="white",
                showgrid=False,
                showspikes=False,
                title="y [mm]",
            ),
        ),
    )

    app.layout = dbc.Container(
        fluid=True,
        style={"padding": "1.5em"},
        children=[
            dcc.Store(id="filename", storage_type="session"),
            dcc.Store(id="event-id", storage_type="session"),
            dcc.Store(id="event-dividers", storage_type="session"),
            html.H1(children="LArPix event display"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            du.Upload(
                                id="select-file",
                                text="Drag and drop or click here to upload",
                                max_file_size=10000,
                                chunk_size=50,
                                default_style={
                                    "width": "15em",
                                    "padding": "0",
                                    "margin": "0",
                                },
                                pause_button=True,
                                filetypes=["h5"],
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.P(
                                children="",
                                id="filename-text",
                                style={"margin": "1em 0 0 0"},
                            ),
                            html.P(
                                children="Event: ",
                                style={
                                    "display": "inline-block",
                                    "font-weight": "bold",
                                },
                            ),
                            dcc.Input(
                                id="input-evid",
                                type="number",
                                placeholder="0",
                                value="0",
                                debounce=True,
                                style={
                                    "width": "5em",
                                    "display": "inline-block",
                                    "margin-right": "0.5em",
                                    "margin-left": "0.5em",
                                },
                            ),
                            html.Div(
                                id="total-events",
                                style={
                                    "padding-right": "1em",
                                    "display": "inline-block",
                                    "text-align": "center",
                                },
                            ),
                        ],
                        width=7,
                    ),
                    dbc.Col(
                        [
                            html.Img(
                                src="https://github.com/soleti/larnd-display/raw/main/docs/logo.png",
                                style={"height": "8em"},
                            )
                        ],
                        width=3,
                        style={"text-align": "right"},
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Alert(
                        "You have reached the end of the file",
                        id="alert-auto",
                        is_open=False,
                        color="warning",
                    ),
                ],
                style={"padding-left": "0.2em"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="event-display",
                                figure=fig,
                                clear_on_unhover=True,
                                style={"height": "85vh"},
                            )
                        ],
                        width=7,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P(id="object-information"),
                                    dcc.Graph(
                                        id="time-histogram", style={"display": "none"}
                                    ),
                                ]
                            )
                        ],
                        width=5,
                    ),
                ]
            ),
        ],
    )

    app.run_server(port=8000, host="127.0.0.1")

    return app


@atexit.register
def clean_cache():
    """Delete uploaded files"""
    try:
        print("Cleaning cache...")
        shutil.rmtree(UPLOAD_FOLDER_ROOT)
    except OSError as err:
        print("Error: %s : %s" % (UPLOAD_FOLDER_ROOT, err.strerror))


if __name__ == "__main__":
    du.configure_upload(app, UPLOAD_FOLDER_ROOT)
    fire.Fire(run_display)
