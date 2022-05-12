#!/usr/bin/env python3

"""
Web-based event display for ArgonCube detectors
"""

from multiprocessing.sharedctypes import Value
import shutil
import atexit

from os.path import basename
from pathlib import Path

import fire
import h5py
import numpy as np

import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import no_update
from dash import dcc
from dash import html
import dash_daq as daq
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, DashProxy, Input, State, MultiplexerTransform

import plotly.graph_objects as go
from plotly import subplots

from larnd_display.display_utils import (
    DetectorGeometry,
    plot_geometry,
    plot_hits,
    plot_light,
    plot_tracks,
)

GEOMETRIES = {}
UPLOAD_FOLDER_ROOT = "cache"
DOCKER_MOUNTED_FOLDER = "/mnt/data/"
CORI_FOLDER = "https://portal.nersc.gov/project/dune/data/"
EVENT_BUFFER = 2000 # maximum difference in timeticks between hit and trigger

app = DashProxy(
    prevent_initial_callbacks=True,
    transforms=[MultiplexerTransform()],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML",
        # "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
    ],
    title="LArPix event display",
)


def draw_event(filename, geometry, event_dividers, event_id, do_plot_tracks, do_plot_opids):
    """Draw 3D event display of event"""
    with h5py.File(filename, "r") as datalog:
        packets = datalog["packets"]

        start_packet = event_dividers[event_id]
        end_packet = event_dividers[event_id + 1]

        last_trigger = packets[start_packet]["timestamp"]
        event_packets = packets[start_packet:end_packet]
        within_buffer = event_packets["timestamp"] - last_trigger < EVENT_BUFFER
        event_packets = event_packets[within_buffer]
        drawn_objects = plot_hits(
            geometry, event_packets, start_packet, last_trigger
        )

        if do_plot_tracks and "mc_packets_assn" in datalog.keys() and "tracks" in datalog.keys():
            tracks = datalog["tracks"]
            mc_packets = datalog["mc_packets_assn"]
            track_ids = np.unique(mc_packets[start_packet:end_packet][within_buffer]["track_ids"])[1:]
            if len(track_ids) > 0:
                drawn_objects.extend(
                    plot_tracks(tracks, range(track_ids[0], track_ids[-1]), event_id)
                )

        drawn_objects.extend(plot_geometry(geometry))

        if "light_trig" in datalog.keys() and do_plot_opids:
            start_t = packets[start_packet]["timestamp"]
            light_indeces = np.argwhere(datalog["light_trig"]["ts_sync"] == start_t)
            integrals = []
            indeces = []
            for light_index in light_indeces:
                waveforms = datalog['light_wvfm'][light_index[0]]
                try:
                    indeces.append(datalog['light_trig'][light_index[0]]['op_channel'])
                    integrals.append(np.sum(waveforms, axis=1))
                except ValueError:
                    break

            if integrals:
                min_integral = min([min(i) for i in integrals])

                for integral, index in zip(integrals, indeces):
                    drawn_objects.extend(
                        plot_light(
                            geometry, integral, index, -min_integral
                        )
                    )

        return drawn_objects

@app.callback(
    [
        Output("event-display", "figure"),
        Output("event-display", "style"),
        Output("alert-geometry", "is_open"),
        Output("alert-geometry", "children"),
        Output("unique-url","children"),
    ],
    [
        Input("event-id", "data"),
        Input("event-dividers", "data"),
        Input("filename", "data"),
        Input("geometry-state", "data"),
        Input("plot-tracks-state", "data"),
        Input("plot-opids-state", "data")
    ],
    [
        State("event-display", "figure"),
    ],
)
def update_output(event_id, event_dividers, filename, geometry, do_plot_tracks, do_plot_opids, figure):
    """Update 3D event display end event id"""
    fig = go.Figure(figure)

    if event_dividers is None:
        return no_update, no_update, no_update, no_update, no_update

    try:
        fig.data = []
        fig.add_traces(draw_event(filename, GEOMETRIES[geometry], event_dividers, event_id, do_plot_tracks, do_plot_opids))
    except IndexError as err:
        print("IndexError",err)
        return fig, {"display": "none"}, True, no_update, no_update
    except KeyError as err:
        print("KeyError",err)
        return fig, {"display": "none"}, True, "Select a geometry first", no_update

    url_filename = filename.replace(DOCKER_MOUNTED_FOLDER, "")
    return fig, {"height": "85vh"}, False, no_update, f"https://larnddisplay.lbl.gov/{url_filename}?geom={geometry}#{event_id}"


@app.callback(Output("light-waveform", "style"), Input("event-id", "data"))
def reset_light(_):
    """Clean the light waveform plot when changing event"""
    return {"display": "none"}


@app.callback(
    [
        Output("light-waveform", "figure"),
        Output("light-waveform", "style")
    ],
    Input("event-display", "clickData"),
    [
        State("event-display", "figure"),
        State("event-id", "data"),
        State("event-dividers", "data"),
        State("filename", "data"),
    ],
)
def light_waveform(click_data, _, event_id, event_dividers, filename):
    """Plot the light waveform for the selected event on the clicked optical detector"""
    if (
        click_data
        and "id" in click_data["points"][0]
        and "opid" in click_data["points"][0]["id"]
    ):
        opid = int(click_data["points"][0]["id"].split("_")[1])
        start_packet = event_dividers[event_id]

        with h5py.File(filename, "r") as datalog:
            if "light_wvfm" not in datalog.keys():
                return go.Figure(), dict(display="none")

            packets = datalog["packets"]
            start_t = packets[start_packet]["timestamp"]
            light_indeces = np.argwhere(datalog["light_trig"]["ts_sync"] == start_t)

            for light_index in light_indeces:
                try:
                    if opid not in datalog['light_trig'][light_index[0]]['op_channel']:
                        continue
                    opid_index = np.argwhere(datalog['light_trig'][light_index[0]]['op_channel']==opid)[0][0]
                except ValueError:
                    opid_index = opid
                except IndexError:
                    print(opid, datalog['light_trig'][light_index[0]]['op_channel'])

                fig = go.Figure(
                    go.Scatter(
                        x=np.arange(0, 256), y=datalog['light_wvfm'][light_index[0]][opid_index]
                    ),
                    layout=dict(
                        title=f"Optical detector {opid}",
                        margin=dict(l=0, r=0, t=60),
                        showlegend=False,
                        template="plotly_white",
                    ),
                )

            fig.update_xaxes(
                linecolor="lightgray",
                matches="x",
                mirror=True,
                ticks="outside",
                showline=True,
            )
            fig.update_yaxes(
                linecolor="lightgray",
                matches="y",
                mirror=True,
                ticks="outside",
                showline=True,
            )

            return fig, dict(display="block", height="300px")

    return no_update, no_update


@app.callback(
    [
        Output("object-information", "children"),
        Output("time-histogram", "figure"),
        Output("time-histogram", "style"),
    ],
    [
        Input("event-id", "data"),
        Input("event-dividers", "data"),
        Input("filename", "data"),
        Input("geometry-detector", "value")
    ],
)
def adc_histogram(event_id, event_dividers, filename, geometry):
    """Plot histogram of the adc counts for each drift volume"""
    if event_dividers is not None and geometry is not None:
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
                tile_id = GEOMETRIES[geometry].get_tile_id(this_io_group, io_channel)
                if (
                    tile_id in GEOMETRIES[geometry].tile_map[0][0]
                    or tile_id in GEOMETRIES[geometry].tile_map[0][1]
                ):
                    anodes.append(0)
                else:
                    anodes.append(1)

            anodes = np.array(anodes)

            n_modules = len(GEOMETRIES[geometry].module_to_io_groups.keys())
            start_t = packets[start_packet]["timestamp"]
            active_modules = []

            for module_id in range(n_modules):
                query = (event_packets["io_group"] - 1) // 4 == module_id
                if len(event_packets[query]) == 0:
                    continue

                active_modules.append(module_id)

            histos = None
            if active_modules:
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

            for i_module_id, module_id in enumerate(active_modules):
                query = (event_packets["io_group"] - 1) // 4 == module_id

                histo1 = go.Histogram(
                    x=event_packets["timestamp"][(anodes == 0) & query] - start_t,
                    xbins=dict(start=0, end=3200, size=20),
                )
                histo2 = go.Histogram(
                    x=event_packets["timestamp"][(anodes == 1) & query] - start_t,
                    xbins=dict(start=0, end=3200, size=20),
                )
                if histos:
                    histos.append_trace(histo1, i_module_id + 1, 1)
                    histos.append_trace(histo2, i_module_id + 1, 2)

            if histos:
                histos.update_annotations(font_size=12)
                histos.update_layout(
                    margin=dict(l=0, r=0, t=50, b=60),
                    showlegend=False,
                    template="plotly_white",
                    title_text="ADC histograms",
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

    return no_update, no_update, no_update


@app.callback(
    Output("filename-text", "children"),
    Input("filename", "modified_timestamp"),
    State("filename", "data"),
)
def update_filename(modified_timestamp, filename):
    """Update the filename text"""
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
    """Update the total number of events text"""
    if modified_timestamp is None:
        raise PreventUpdate

    if not event_dividers:
        total_events = 0
    else:
        total_events = len(event_dividers) - 2

    return f"/ {total_events}"

@app.callback(
    [
        Input("input-evid", "value"),
        Input("event-dividers", "data"),
    ],
    Output("event-id", "data"),
)
def update_event_id_click(input_evid, event_dividers):
    try:
        event_id = int(input_evid)
    except TypeError:
        return no_update

    if event_dividers:
        if event_id >= len(event_dividers) - 1:
            event_id = len(event_dividers) - 2

        if event_id < 0:
            event_id = 0

        return event_id
    else:
        return no_update

@app.callback(
    Output("input-evid", "value"),
    Input("event-id", "modified_timestamp"),
    State("event-id", "data"),
)
def update_event_id(modified_timestamp, event_id):
    """Update the event id input box with the current event id"""
    if modified_timestamp is None:
        raise PreventUpdate

    try:
        return int(event_id)
    except TypeError:
        raise PreventUpdate

@app.callback(
    Output("geometry-detector", "value"),
    Input("geometry-state", "modified_timestamp"),
    State("geometry-state", "data"),
)
def update_geometry(modified_timestamp, detector_geometry):
    """Update the event id input box with the current event id"""
    if modified_timestamp is None:
        raise PreventUpdate

    try:
        return detector_geometry
    except TypeError:
        raise PreventUpdate

@app.callback(
    [
        Output("filename", "data"),
        Output("event-dividers", "data"),
        Output("event-id", "data"),
        Output("alert-file-not-found", "is_open"),
        Output("alert-file-not-found", "children"),
    ],
    Input("server-filename", "value"),
    [
        State("event-id", "data"),
        State("filepath", "children"),
    ],
)
def select_file(input_filename, event_id, filepath):
    filepath_message = filepath
    if event_id is None:
        event_id = 0

    try:
        try:
            if filepath['props']['children'] == CORI_FOLDER:
                load_filepath = DOCKER_MOUNTED_FOLDER
                filepath_message = CORI_FOLDER
        except TypeError:
            load_filepath = filepath

        if len(input_filename) > 0 and input_filename[0] == "/":
            input_filename = input_filename[1:]

        h5_file = Path(load_filepath) / input_filename
        datalog = h5py.File(h5_file, "r")
    except FileNotFoundError:
        print(h5_file, "not found")
        return no_update, no_update, event_id, True, f"File {filepath_message}{input_filename} not found"
    except IsADirectoryError:
        return no_update, no_update, event_id, False, ""
    except OSError as err:
        print(h5_file, "invalid file", err)
        return no_update, no_update, event_id, True, f"File {filepath_message}{input_filename} is not a valid file"

    packets = datalog["packets"]

    trigger_packets = np.argwhere(packets["packet_type"] == 7).T[0]
    if trigger_packets.size > 0:
        event_dividers = trigger_packets[:-1][np.diff(trigger_packets) != 1]
        event_dividers = np.append(event_dividers, [trigger_packets[-1], len(packets)])

        return str(h5_file), event_dividers, event_id, False, ""
    else:
        return no_update, no_update, event_id, True, "No triggers found"

@app.callback(
    [
        Output("filename", "data"),
        Output("event-dividers", "data"),
        Output("event-id", "data"),
        Output("server-filename", "value")
    ],
    Input("select-file", "isCompleted"),
    [
        State("filename", "data"),
        State("event-dividers", "data"),
        State("select-file", "fileNames"),
        State("select-file", "upload_id"),
    ],
)
def upload_file(is_completed, filename, event_dividers, filenames, upload_id):
    """Upload HDF5 file to cache"""
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

        return str(h5_file), event_dividers, 0, ""

    return filename, event_dividers, 0, ""


@app.callback(
    Output('geometry-state', 'data'),
    Input('geometry-detector', 'value'),
)
def update_geometry_state(geometry_detector):
    return geometry_detector


@app.callback(
    Output("plot-tracks-state", "data"),
    Input("plot-tracks", "on")
)
def set_plot_true_tracks(on):
    return on


@app.callback(
    Output("plot-opids-state", "data"),
    Input("plot-opids", "on")
)
def set_plot_opids(on):
    return on

@app.callback(
    [
        Output("server-filename", "value"),
        Output("event-id", "data"),
        Output("geometry-state", "data"),
    ],
    [
        Input('url', 'pathname'),
        Input('url', 'hash'),
        Input('url', 'search')
    ]
)
def display_page(pathname, this_hash, search):
    if pathname[1:] and "?geom=" in search:
        if this_hash:
            try:
                event_id = int(this_hash[1:])
            except ValueError:
                event_id = 0
        else:
            event_id = 0

        geom = search.split("?geom=")[1]

        return pathname[1:], event_id, geom

    return no_update, no_update, no_update


def run_display(larndsim_dir, host="127.0.0.1", port=5000, filepath="."):
    """Create layout and run Dash app"""
    global GEOMETRIES

    if filepath[-1] != "/":
        filepath += "/"

    module0_detector_properties = larndsim_dir + "/larndsim/detector_properties/module0.yaml"
    twobytwo_detector_properties = larndsim_dir + "/larndsim/detector_properties/2x2.yaml"
    tile44_layout = larndsim_dir + "/larndsim/pixel_layouts/multi_tile_layout-2.3.16.yaml"

    GEOMETRIES["Module-0"] = DetectorGeometry(module0_detector_properties, tile44_layout)
    GEOMETRIES["2x2"] = DetectorGeometry(twobytwo_detector_properties, tile44_layout)

    fig = go.Figure()
    camera = dict(eye=dict(x=-2, y=0.3, z=1.1), center=dict(x=0, y=0, z=-0.2),)

    fig.update_layout(
        scene_camera=camera,
        uirevision=True,
        margin=dict(l=0, r=0, t=4),
        legend={"x": 0},
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
        style={
            "padding": "1.5em",
            "background-image": "url('%s')" % app.get_asset_url("logo.png"),
            "background-size": "74px 54px",
            "background-position": "right 1em top 1em",
            "background-repeat": "no-repeat",
        },
        children=[
            dcc.Location(id='url'),
            dcc.Store(id="filename", storage_type="session"),
            dcc.Store(id="event-id", storage_type="session"),
            dcc.Store(id="event-dividers", storage_type="session"),
            dcc.Store(id="geometry-state", storage_type="session"),
            dcc.Store(id="plot-tracks-state", storage_type="session", data=False),
            dcc.Store(id="plot-opids-state", storage_type="session", data=False),
            html.Div(id="unique-url", style={"display": "none"}),
            html.P(id='test'),
            html.H1(children="LArPix event display"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            du.Upload(
                                id="select-file",
                                text="Drag and drop or click here to upload",
                                max_file_size=10000,
                                chunk_size=0.5,
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
                                children=["Or specify existing file path here:"],
                                style={"margin": "0"},
                            ),
                            html.P(
                                children=html.A(href=filepath, children=filepath) if "https" in filepath else filepath,
                                id="filepath",
                                style={
                                    "display": "inline-block",
                                    "font-family": "monospace",
                                    "font-size": "small",
                                    "margin": "0 0 0.2em 0",
                                },
                            ),
                            dcc.Input(
                                id="server-filename",
                                type="text",
                                style={
                                    "font-family": "monospace",
                                    "font-size": "small",
                                    "margin": "0 0 0.2em 0",
                                },
                                placeholder="enter file path here...",
                                size=38,
                                debounce=True
                            ),
                            html.P(
                                children=[
                                    html.Span("Detector geometry: ", style={"vertical-align": "0.6em"}),
                                    dcc.Dropdown(
                                        id='geometry-detector',
                                        options=[{'label': x, 'value': x} for x in ['Module-0', '2x2']],
                                        value='Module-0',
                                        style={"width": "10em", "display": "inline-block", "height": "2.4em"},
                                    )
                                ], style={"margin": "0"}
                            ),
                            html.P(
                                children=[
                                    "Copy unique URL to clipboard ",
                                    dcc.Clipboard(
                                        target_id="unique-url",
                                        title="copy",
                                        style={
                                            "display": "inline-block",
                                            "fontSize": 20,
                                            "verticalAlign": "top",
                                        },
                                    )
                                ],
                                style={"margin": "0"}
                            ),
                            dbc.Alert(
                                children=["File not found"],
                                id="alert-file-not-found",
                                is_open=False,
                                duration=3000,
                                color="warning",
                            ),
                            dbc.Alert(
                                "Error loading the file with this detector geometry",
                                id="alert-geometry",
                                duration=3000,
                                is_open=False,
                                color="danger",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.P(
                                children="",
                                id="filename-text",
                                style={"margin": "0"},
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
                            html.P(
                                children=[
                                    "Plot true tracks ",
                                    daq.BooleanSwitch(id='plot-tracks', on=False, style={"display": "inline-block"})
                                ],
                                style={"margin": "0"}
                            ),
                            html.P(
                                children=[
                                    "Plot optical detectors ",
                                    daq.BooleanSwitch(id='plot-opids', on=False, style={"display": "inline-block"})
                                ],
                                style={"margin": "0"}
                            ),
                        ],
                        width=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Loading(dcc.Graph(
                                id="event-display",
                                figure=fig,
                                clear_on_unhover=True,
                                style={"display": "none"},
                                config = {
                                    'toImageButtonOptions': {
                                        'format': 'png', # one of png, svg, jpeg, webp
                                        'height': 900,
                                        'width': 1200,
                                    },
                                    'displaylogo': False
                                }
                            ))
                        ],
                        width=7,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P(id="object-information"),
                                    dcc.Graph(
                                        id="time-histogram", style={"display": "none"},
                                        config = {'displaylogo': False}
                                    ),
                                    dcc.Graph(
                                        id="light-waveform", style={"display": "none"},
                                        config = {'displaylogo': False}
                                    ),
                                ]
                            )
                        ],
                        width=5,
                    ),
                ]
            ),
            dbc.Row([
                html.Footer(
                    children=[
                        "Developed by ",
                        html.A("Stefano Roberto Soleti", href="mailto:roberto@lbl.gov"),
                        " for the DUNE collaboration. Open issues and send suggestions on ",
                        html.A("GitHub", href="https://github.com/soleti/larnd-display"),
                        "."
                    ], style={'font-size':'x-small'}
                )
            ])
        ],
    )

    app.run_server(port=port, host=host)

    return app


@atexit.register
def clean_cache():
    """Delete uploaded files"""
    try:
        print("Cleaning cache...")
        shutil.rmtree(UPLOAD_FOLDER_ROOT)
    except OSError as err:
        print("Can't clean %s : %s" % (UPLOAD_FOLDER_ROOT, err.strerror))


if __name__ == "__main__":
    du.configure_upload(app, UPLOAD_FOLDER_ROOT)
    fire.Fire(run_display)
