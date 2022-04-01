#!/usr/bin/env python3

import h5py
from matplotlib.pyplot import draw
import numpy as np

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
from plotly import subplots

from larnd_display.display_utils import DetectorGeometry, plot_geometry, plot_hits, plot_light, plot_tracks
from larndsim.consts import detector

fig = go.Figure(plot_geometry())
camera = dict(
    eye=dict(x=-2,y=0.3,z=1.1)
)

fig.update_layout(scene_camera=camera,
                  uirevision=True,
                  margin=dict(l=0, r=0, t=5),
                  legend={"y" : 0.8},
                  scene=dict(
                      xaxis=dict(backgroundcolor="white",
                                 showspikes=False,
                                 showgrid=False,
                                 title='x [mm]'),
                                #  range=(detector.TPC_BORDERS[0][0][0],detector.TPC_BORDERS[0][0][1])),
                      yaxis=dict(backgroundcolor="white",
                                 showgrid=False,
                                 showspikes=False,
                                 title='z [mm]'),
                                #  range=(detector.TPC_BORDERS[0][2][0],detector.TPC_BORDERS[1][2][0])),
                      zaxis=dict(backgroundcolor="white",
                                 showgrid=False,
                                 showspikes=False,
                                 title='y [mm]'),
                                #  range=(detector.TPC_BORDERS[0][1][0],detector.TPC_BORDERS[0][1][1])),
                  ))

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
                external_scripts=[
                    'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML',
                ],
                title='LArND event display')

evid = 0

@app.callback(
    [Output("event-display", "figure"),
     Output("alert-auto", "is_open"),
     Output('input-evid', 'value')],
    [Input('following-button', 'n_clicks'),
     Input('previous-button', 'n_clicks'),
     Input("input-evid", "value")],
     State('event-display', 'figure'))
def update_output(n_clicks_next, n_clicks_prev, n_events, figure):
    global evid

    following = 'following-button.n_clicks' in [p['prop_id'] for p in dash.callback_context.triggered][0]
    previous = 'previous-button.n_clicks' in [p['prop_id'] for p in dash.callback_context.triggered][0]

    fig = go.Figure(figure)

    try:
        n_events = int(n_events)
    except TypeError:
        n_events = evid
        return fig, True, n_events


    if following:
        n_events += 1

    if previous:
        n_events -= 1

    if n_events >= len(event_dividers)-1:
        n_events = len(event_dividers)-2
        return fig, True, n_events
    if n_events < 0:
        n_events = 0
        return fig, True, 0

    if evid != n_events:
        evid = n_events
        start_packet = event_dividers[n_events]
        end_packet = event_dividers[n_events+1]

        track_ids = np.unique(mc_packets[start_packet:end_packet]['track_ids'])[1:]
        last_trigger = packets[start_packet]['timestamp']
        event_packets = packets[start_packet:end_packet]
        drawn_objects = plot_hits(my_geometry, event_packets, start_packet, last_trigger)
        drawn_objects.extend(plot_tracks(tracks, range(track_ids[0], track_ids[-1]), n_events))
        drawn_objects.extend(plot_geometry())
        if light_lut:
            drawn_objects.extend(plot_light(my_geometry,np.sum(light_lut[track_ids]['n_photons_det'],axis=0)))
        fig.data = []
        fig.add_traces(drawn_objects)

    # if click_data:
    #     if 'customdata' in click_data['points'][0]:
    #         if 'point' in click_data['points'][0]['customdata']:
    #             packet_id = int(click_data['points'][0]['customdata'].split('_')[1])
    #             track_ids = mc_packets[packet_id]['track_ids']
    #             fractions = mc_packets[packet_id]['fraction']
    #             track_ids = [t for t,f in zip(track_ids,fractions) if f > 0]
    #             output_track_ids = ', '.join(['%i' % i for i in track_ids])
    #             output_fractions = ', '.join(['%f' % f for f in fractions if f > 0])
    #             for itr,trace in enumerate(fig.data):
    #                 if trace['customdata']:
    #                     if "track" in trace['customdata'][0]:
    #                         track_id = int(trace['customdata'][0].split('_')[1])
    #                         if track_id in track_ids:
    #                             fig.data[itr]['line']['width'] = 18
    #                             fig.data[itr]['opacity'] = 0.9
    #                         elif fig.data[itr]['line']['width'] == 18:
    #                             fig.data[itr]['line']['width'] = 12
    #                             fig.data[itr]['opacity'] = 0.7

    #             output = 'Hit associated to tracks ' + output_track_ids + ' with fractions ' + output_fractions

    return fig, False, n_events

@app.callback(
    [Output('object-information', 'children'),
     Output('time-histogram', 'figure'),
     Output('time-histogram', 'style')],
    [Input('following-button', 'n_clicks'),
     Input('input-evid', 'value')])
def histogram(n_clicks, n_events):
    start_packet = event_dividers[n_events]
    end_packet = event_dividers[n_events+1]
    event_packets = packets[start_packet:end_packet]
    event_packets = event_packets[event_packets['packet_type']==0]
    anodes = []

    for io_group,io_channel in zip(event_packets['io_group'],event_packets['io_channel']):
        if not io_group % 4:
            this_io_group = 4
        else:
            this_io_group = io_group % 4
        tile_id = my_geometry.get_tile_id(this_io_group,io_channel)
        if tile_id in detector.TILE_MAP[0][0] or tile_id in detector.TILE_MAP[0][1]:
            anodes.append(0)
        else:
            anodes.append(1)

    anodes = np.array(anodes)
    start_t = packets[tr][::2][n_events]['timestamp']

    n_modules = len(detector.MODULE_TO_IO_GROUPS.keys())

    active_modules = []

    for module_id in range(n_modules):
        query = ((event_packets['io_group']-1)//4==module_id)
        if len(event_packets[query]) == 0:
            continue

        active_modules.append(module_id)

    histos = subplots.make_subplots(rows=len(active_modules), cols=2,
                                    subplot_titles=['(%i,%i)' % (m+1,p+1) for m in active_modules for p in range(2)],
                                    vertical_spacing=0.25/len(active_modules) if len(active_modules) else 0,
                                    shared_xaxes=True, shared_yaxes=True)

    for im, module_id in enumerate(active_modules):
        query = ((event_packets['io_group']-1)//4==module_id)

        histo1 = go.Histogram(
            x=event_packets['timestamp'][(anodes==0) & query],
            xbins=dict(
                start=start_t,
                end=start_t+3200,
                size=20
            ),
        )
        histo2 = go.Histogram(
            x=event_packets['timestamp'][(anodes==1) & query],
            xbins=dict(
                start=start_t,
                end=start_t+3200,
                size=20
            ),
        )

        histos.append_trace(histo1, im+1, 1)
        histos.append_trace(histo2, im+1, 2)

    histos.update_annotations(font_size=12)
    histos.update_layout(margin=dict(l=0, r=0, t=30, b=10),showlegend=False,template='plotly_white')
    histos.update_xaxes(title_text="Time [timestamp]", row=len(active_modules))
    histos.update_xaxes(linecolor='lightgray',matches='x',mirror=True,ticks='outside',showline=True)
    histos.update_yaxes(linecolor='lightgray',matches='y',mirror=True,ticks='outside',showline=True)
    subplots_height =  "%fvh" % (len(active_modules)*18+5)
    return "", histos, dict(height=subplots_height)

event_dividers = None
mc_packets = None
packets = None
light_lut = None
my_geometry = None
tr = None
tracks = None

def run_display(input_file, detector_properties, pixel_layout):
    global event_dividers
    global mc_packets
    global packets
    global light_lut
    global my_geometry
    global tr
    global tracks

    my_geometry = DetectorGeometry(detector_properties, pixel_layout)
    datalog = h5py.File(input_file, 'r')
    tracks = datalog['tracks']
    packets = datalog['packets']
    mc_packets = datalog['mc_packets_assn']
    if 'light_dat' in datalog.keys():
        light_lut = datalog['light_dat']
    tr = packets['packet_type'] == 7
    trigger_packets = np.argwhere(tr).T[0]
    event_dividers = trigger_packets[:-1][np.diff(trigger_packets)!=1]
    event_dividers = np.append(event_dividers,[trigger_packets[-1],len(packets)])
    start_packet = event_dividers[0]
    end_packet = event_dividers[1]
    last_trigger = packets[start_packet]['timestamp']
    event_packets = packets[start_packet:end_packet]
    drawn_objects = plot_hits(my_geometry, event_packets, start_packet, last_trigger)
    track_ids = np.unique(mc_packets[start_packet:end_packet]['track_ids'])[1:]
    drawn_objects.extend(plot_tracks(tracks, track_ids, 0))
    drawn_objects.extend(plot_geometry())
    if light_lut:
        drawn_objects.extend(plot_light(my_geometry,np.sum(light_lut[track_ids]['n_photons_det'],axis=0)))
    fig.add_traces(drawn_objects)

    app.layout = dbc.Container(
        fluid=True,
        style={'margin': '1.5em'},
        children=[
            html.H1(children='LArPix event display'),
            html.P(children=[
                'Input file: ',
                html.Span(input_file, style={'font-family': 'monospace'}),
                html.Br(),
                'Detector properties: ',
                html.Span(detector_properties, style={'font-family': 'monospace'}),
                html.Br(),
                'Pixel layout: ',
                html.Span(pixel_layout, style={'font-family': 'monospace'}),
            ]),
            dbc.Row([
                html.Div([
                    dbc.Button('<', id='previous-button', n_clicks=0, color="primary"),
                    dcc.Input(id="input-evid",
                              type='number',
                              placeholder="0",
                              value="0",
                              debounce=True,
                              style={'width': '5em', 'display': 'inline-block', 'margin-right': '0.5em', 'margin-left': '0.5em'}),
                    html.Div(children=f'/{len(event_dividers)-2}', style={'margin-right': '1em','display':'inline-block','text-align':'center'}),
                    dbc.Button('>', id='following-button', n_clicks=0, color="primary")
                ]),
            ],style={'margin-left': '0.2em','margin-bottom':'1em'}),
            dbc.Row([
                dbc.Alert(
                        "You have reached the end of the file",
                        id="alert-auto",
                        is_open=False,
                        color="warning"
                ),
            ],style={'margin-left': '0.2em'}),
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(id='event-display',
                                       figure=fig,
                                       clear_on_unhover=True,
                                       style={'height': '90vh'})],
                            width=9),
                    dbc.Col([html.H4(children='ADC timestamp histograms'),
                             html.Div([html.P(id='object-information'),
                                       dcc.Graph(id='time-histogram')],
                                       style={'margin-right':'2em'})],
                            width=3),
                ]
            )
        ]
    )

    app.run_server(port=8000, host='127.0.0.1')

    return app

import fire
if __name__ == "__main__":
    fire.Fire(run_display)

