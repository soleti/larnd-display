#!/usr/bin/env python3

from collections import defaultdict

import fire
import h5py
import numpy as np
from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, no_update
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from larnd_display.display_utils import DetectorGeometry
from larndsim.consts import detector

MY_GEOMETRY = None

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.DARKLY], title="ND-Lar event display"
)

@app.callback(
    [
        Output("event-display-zy", "figure"),
        Output("event-display-zy", "style"),
        Output("event-display-xy", "figure"),
        Output("event-display-xy", "style"),
        Output("event-display-xz", "figure"),
        Output("event-display-xz", "style"),
        Output("alert-file-not-found", "is_open"),
        Output("alert-file-not-found", "children"),
    ],
    Input("submit-val", "n_clicks"),
    State("input_filename", "value"),
)
def draw_event(_, filename):

    if filename is None:
        raise PreventUpdate

    points = defaultdict(lambda: defaultdict(lambda: 0))
    points_xy = defaultdict(lambda: 0)
    points_xz = defaultdict(lambda: 0)

    try:
        print(filename)
        datalog = h5py.File(filename)
    except FileNotFoundError:
        print(filename, "not found")
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            True,
            f"File {filename} not found",
        )
    except (IsADirectoryError, OSError) as err:
        print(filename, "invalid file", err)
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            True,
            f"File {filename} is not a valid HDF5 file",
        )

    packets = datalog["packets"]
    event_packets = packets
    x_borders = np.unique(detector.TPC_BORDERS[:, 0, :], axis=0) * 10
    z_borders = (
        np.array(
            [
                [-255.9, -254.6],
                [-153.8, -152.5],
                [-51.7, -50.4],
                [50.4, 51.7],
                [152.5, 153.8],
                [254.6, 255.9],
            ]
        )
        * 10
    )

    last_trigger = 0
    for packet in tqdm(event_packets):
        if packet["packet_type"] == 7:
            last_trigger = packet["timestamp"]
            continue
        if packet["packet_type"] != 0:
            continue
        io_group, io_channel, chip, channel, time = (
            packet["io_group"],
            packet["io_channel"],
            packet["chip_id"],
            packet["channel_id"],
            packet["timestamp"],
        )
        module_id = (io_group - 1) // 4
        if packet["timestamp"] - last_trigger > 1e6:
            last_trigger = packet["timestamp"]

        io_group = io_group - (io_group - 1) // 4 * 4
        z = MY_GEOMETRY.get_z_coordinate(io_group, io_channel, time - last_trigger)
        x, y = MY_GEOMETRY.geometry[(io_group, io_channel, chip, channel)]

        x += MY_GEOMETRY.tpc_offsets[module_id][2] * 10
        z += MY_GEOMETRY.tpc_offsets[module_id][0] * 10

        for im_x, x_border in enumerate(x_borders):
            if x_border[0] < x < x_border[1]:
                points[im_x][(x, y)] += packet["dataword"]

        points_xy[(z, y)] += packet["dataword"]
        points_xz[(z, x)] += packet["dataword"]

    fig_xy = go.Figure(
        layout={
            "margin": dict(l=10, r=10, t=10, b=10),
            "xaxis": dict(showgrid=False),
            "yaxis": dict(showgrid=False),
            "template": "plotly_dark",
        }
    )

    a_points = np.array(list(points_xy.keys()))
    a_charge = np.array(list(points_xy.values()))
    pixel_pitch = 3.8
    y_range = [detector.TPC_BORDERS[0][1][0] * 10, detector.TPC_BORDERS[0][1][1] * 10]

    fig_xy.add_trace(
        go.Histogram2d(
            x=a_points[:, 0][a_charge > 0],
            y=a_points[:, 1][a_charge > 0],
            z=np.log10(a_charge[a_charge > 0]),
            autobinx=False,
            xbins=dict(size=detector.V_DRIFT * detector.TIME_SAMPLING * 15 * 20),
            autobiny=False,
            ybins=dict(start=y_range[0], end=y_range[1], size=pixel_pitch),
            zmin=0,
            hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>log10(ADC): %{z:.2f} <extra></extra>",
            histfunc="avg",
            colorscale="jet",
            colorbar={"title": "log10(ADC)"},
            zmax=np.log10(2000),
        )
    )
    fig_xy.update_yaxes(title="y [mm]", scaleanchor="x", scaleratio=1, zeroline=False)
    fig_xy.update_xaxes(title="x [mm]", zeroline=False)

    for border in z_borders:
        fig_xy.add_shape(
            type="rect",
            x0=border[0],
            y0=y_range[0],
            x1=border[1],
            y1=y_range[1],
            fillcolor="rgb(17,17,17)",
            line=dict(
                width=0,
            ),
        )

    x_range = [4139.1, 9164.9]

    fig_xz = go.Figure(
        layout={
            "margin": dict(l=10, r=10, t=10, b=10),
            "xaxis": dict(showgrid=False),
            "yaxis": dict(showgrid=False),
            "template": "plotly_dark",
        }
    )

    a_points = np.array(list(points_xz.keys()))
    a_charge = np.array(list(points_xz.values()))
    pixel_pitch = 3.8

    fig_xz.add_trace(
        go.Histogram2d(
            x=a_points[:, 0][a_charge > 0],
            y=a_points[:, 1][a_charge > 0],
            z=np.log10(a_charge[a_charge > 0]),
            autobinx=False,
            xbins=dict(size=detector.V_DRIFT * detector.TIME_SAMPLING * 15 * 20),
            autobiny=False,
            ybins=dict(size=pixel_pitch),
            zmin=0,
            hovertemplate="z: %{x:.1f}<br>x: %{y:.1f}<br>log10(ADC): %{z:.2f} <extra></extra>",
            histfunc="avg",
            colorscale="jet",
            colorbar={"title": "log10(ADC)"},
            zmax=np.log10(2000),
        )
    )
    fig_xz.update_yaxes(title="z [mm]", scaleanchor="x", scaleratio=1, zeroline=False)
    fig_xz.update_xaxes(title="x [mm]", zeroline=False)
    z_range = (
        np.min(detector.TPC_BORDERS[:, 2, :]) * 10,
        np.max(detector.TPC_BORDERS[:, 2, :]) * 10,
    )
    for border in z_borders:
        fig_xz.add_shape(
            type="rect",
            x0=border[0],
            y0=x_range[0],
            x1=border[1],
            y1=x_range[1],
            fillcolor="rgb(17,17,17)",
            line=dict(
                width=0,
            ),
        )
    x_edges = np.array(
        [[5110, 5151.4], [6124.6, 6165.6], [7138.4, 7179.8], [8152.6, 8194]]
    )
    for border in x_edges:
        fig_xz.add_shape(
            type="rect",
            x0=z_range[0],
            y0=border[0],
            x1=z_range[1],
            y1=border[1],
            fillcolor="rgb(17,17,17)",
            line=dict(
                width=0,
            ),
        )

    fig_zy = make_subplots(
        rows=1, cols=5, horizontal_spacing=0.005, x_title="z [mm]", y_title="y [mm]"
    )
    for im_x in range(len(x_borders)):
        a_points = np.array(list(points[im_x].keys()))
        a_charge = np.array(list(points[im_x].values()))

        fig_zy.add_trace(
            go.Histogram2d(
                x=a_points[:, 0][a_charge > 0],
                y=a_points[:, 1][a_charge > 0],
                z=np.log10(a_charge[a_charge > 0]),
                autobinx=False,
                xbins=dict(size=pixel_pitch),
                autobiny=False,
                ybins=dict(start=y_range[0], end=y_range[1], size=pixel_pitch),
                zmin=0,
                hovertemplate="z: %{x:.1f}<br>y: %{y:.1f}<br>log10(ADC): %{z:.2f} <extra></extra>",
                histfunc="avg",
                colorscale="jet",
                colorbar={"title": "log10(ADC)"},
                zmax=np.log10(2000),
            ),
            col=im_x + 1,
            row=1,
        )
        fig_zy.update_yaxes(
            showgrid=False,
            zeroline=False,
            row=1,
            col=im_x + 1,
            scaleratio=1,
            visible=im_x == 0,
            scaleanchor="x" if im_x == 0 else f"x{im_x+1}",
        )
        fig_zy.update_xaxes(showgrid=False, zeroline=False, row=1, col=im_x + 1)

    fig_zy.update_layout(
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        template="plotly_dark",
        margin=dict(l=70, r=0, t=20, b=50),
    )

    return (
        fig_zy,
        dict(display="block", height="86vh", width="72vw"),
        fig_xy,
        dict(display="block", height="80vh"),
        fig_xz,
        dict(display="block", height="80vh"),
        False,
        no_update,
    )


config = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "height": 1000 * 4,
        "width": 1700 * 4,
    },
}

app.layout = html.Div(
    children=[
        html.P(
            children=[
                "Input file: ",
                dcc.Input(
                    id="input_filename",
                    type="text",
                    placeholder="Enter file path here...",
                    debounce=True,
                    style={"background-color": "rgb(32, 32, 32)", "color": "white"},
                ),
                html.Button(
                    "Submit", id="submit-val", n_clicks=0, style={"margin-left": "10px"}
                ),
                dbc.Alert(
                    children=["File not found"],
                    id="alert-file-not-found",
                    is_open=False,
                    duration=3000,
                    style={"width": "30vw"},
                    color="warning",
                ),
            ]
        ),
        dbc.Tabs(
            children=[
                dbc.Tab(
                    label="ZY projection",
                    children=[
                        dcc.Loading(
                            dcc.Graph(
                                id="event-display-zy",
                                style={"display": "none"},
                                config=config,
                            )
                        ),
                    ],
                ),
                dbc.Tab(
                    label="XY projection",
                    children=[
                        dcc.Loading(
                            dcc.Graph(
                                id="event-display-xy",
                                style={"display": "none"},
                                config=config,
                            )
                        ),
                    ],
                ),
                dbc.Tab(
                    label="XZ projection",
                    children=[
                        dcc.Loading(
                            dcc.Graph(
                                id="event-display-xz",
                                style={"display": "none"},
                                config=config,
                            )
                        ),
                    ],
                ),
            ],
            style={"width": "100vw"},
        ),
    ],
    style={
        "width": "100vw",
        "height": "100vh",
        "padding": "1em",
        "margin": "0",
        "background-color": "rgb(17,17,17)",
    },
)


def run_display(detector_properties, pixel_layout, host="127.0.0.1", port=8050):
    global MY_GEOMETRY

    MY_GEOMETRY = DetectorGeometry(
        detector_properties,
        pixel_layout,
    )

    app.run_server(port=port, host=host)

if __name__ == "__main__":
    fire.Fire(run_display)