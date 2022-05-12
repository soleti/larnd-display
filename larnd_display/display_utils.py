"""Utilities module used by the event display"""

from collections import defaultdict
from charset_normalizer import detect

import numpy as np
import yaml

from particle import Particle

from larndsim import consts
from larndsim.consts import detector

import plotly.graph_objects as go

import plotly.colors

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")

COLORSCALE = plotly.colors.make_colorscale(plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.YlOrRd)[0])

COLOR_DICT = defaultdict(lambda: 'gray', {11: '#3f90da',
                                          -11: '#92dadd',
                                          13: '#b2df8a',
                                          -13: '#33a02c',
                                          22: '#b15928',
                                          2212: '#bd1f01',
                                          -2212: '#e76300',
                                          -211: '#cab2d6',
                                          211: '#6a3d9a',
                                          2112: '#555555',
                                          1000010020: 'blue'})

def plot_geometry(this_detector):
    """Plot detector geometry"""
    drawn_objects = []

    for ix in range(0,this_detector.tpc_borders.shape[0],2):
        for i in range(2):
            for j in range(2):
                drawn_objects.append(go.Scatter3d(x=(this_detector.tpc_borders[ix][0][j],this_detector.tpc_borders[ix][0][j]),
                                                  y=(this_detector.tpc_borders[ix][2][0],this_detector.tpc_borders[ix+1][2][0]),
                                                  z=(this_detector.tpc_borders[ix][1][i],this_detector.tpc_borders[ix][1][i]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))

                drawn_objects.append(go.Scatter3d(x=(this_detector.tpc_borders[ix][0][j],this_detector.tpc_borders[ix][0][j]),
                                                  y=(this_detector.tpc_borders[ix+i][2][0],this_detector.tpc_borders[ix+i][2][0]),
                                                  z=(this_detector.tpc_borders[ix][1][0],this_detector.tpc_borders[ix][1][1]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))

                drawn_objects.append(go.Scatter3d(x=(this_detector.tpc_borders[ix][0][0],this_detector.tpc_borders[ix][0][1]),
                                                  y=(this_detector.tpc_borders[ix+i][2][0],this_detector.tpc_borders[ix+i][2][0]),
                                                  z=(this_detector.tpc_borders[ix][1][j],this_detector.tpc_borders[ix][1][j]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))

        xx = np.linspace(this_detector.tpc_borders[ix][0][0], this_detector.tpc_borders[ix][0][1], 2)
        zz = np.linspace(this_detector.tpc_borders[ix][1][0], this_detector.tpc_borders[ix][1][1], 2)
        xx,zz = np.meshgrid(xx,zz)

        single_color=[[0.0, 'rgb(200,200,200)'], [1.0, 'rgb(200,200,200)']]
        z_cathode = (this_detector.tpc_borders[ix][2][0]+this_detector.tpc_borders[ix+1][2][0])/2

        cathode_plane=dict(type='surface', x=xx, y=np.full(xx.shape, z_cathode), z=zz,
                           opacity=0.10,
                           hoverinfo='skip',
                           text='Cathode',
                           colorscale=single_color,
                           showlegend=False,
                           showscale=False)

        drawn_objects.append(cathode_plane)

        annotations_x = [(p[0][0]+p[0][1])/2 for p in this_detector.tpc_borders]
        annotations_y = [p[1][1]for p in this_detector.tpc_borders]
        annotations_z = [(p[2][0]+p[2][1])/2 for p in this_detector.tpc_borders]

        annotations_label = ["(%i,%i)" % (ip//2+1,ip%2+1) for ip in range(this_detector.tpc_borders.shape[0])]
        module_annotations = go.Scatter3d(
            mode='text',
            x=annotations_x,
            z=annotations_y,
            y=annotations_z,
            text=annotations_label,
            opacity=0.5,
            textfont=dict(
                color='gray',
                size=16
            ),
            showlegend=False,
        )
        drawn_objects.append(module_annotations)

    return drawn_objects

def plot_light(this_detector, n_photons, op_indeces, max_integral):
    """Plot optical detectors"""
    drawn_objects = []
    ys = np.flip(np.array([-595.43, -545.68, -490.48, -440.73, -385.53, -335.78,
                           -283.65, -236.65, -178.70, -131.70, -73.75, -26.75,
                           25.38, 75.13, 130.33, 180.08, 235.28, 285.03, 337.15,
                           384.15, 442.10, 489.10, 547.05, 594.05])/10)
    light_width = ys[1]-ys[0]

    for ix in range(0,this_detector.tpc_borders.shape[0]):
        for ilight, light_y in enumerate(ys):
            for iside in range(2):

                opid = ilight + iside*len(ys) + ix*len(ys)*2
                if opid not in op_indeces:
                    continue
                xx = np.linspace(this_detector.tpc_borders[ix][2][0], this_detector.tpc_borders[ix][2][1], 2)
                zz = np.linspace(light_y - light_width/2 + this_detector.tpc_offsets[0][1] + 0.25,
                                 light_y + light_width/2 + this_detector.tpc_offsets[0][1] - 0.25, 2)

                xx,zz = np.meshgrid(xx,zz)

                light_color=[[0.0, get_continuous_color(COLORSCALE, intermed=max(0,-n_photons[opid%96])/max_integral)],
                             [1.0, get_continuous_color(COLORSCALE, intermed=max(0,-n_photons[opid%96])/max_integral)]]

                if ix % 2 == 0:
                    flip = 0
                else:
                    flip = -1

                opid_str = f"opid_{opid}"
                light_plane = dict(type='surface', y=xx, x=np.full(xx.shape, this_detector.tpc_borders[ix][0][iside+flip]), z=zz,
                                   opacity=0.15,
                                   hoverinfo='text',
                                   ids=[[opid_str, opid_str], [opid_str, opid_str]],
                                   text=f'Optical detector {opid} waveform integral<br>{n_photons[opid%96]:.2e}',
                                   colorscale=light_color,
                                   showlegend=False,
                                   showscale=False)

                drawn_objects.append(light_plane)

    return drawn_objects

def plot_hits(this_detector, event_packets, start_packet, last_trigger):
    """Plot 3D hits starting from packet information"""
    hits = [[], [], []]
    hits_index = []
    hits_charge = []
    hits_text = []

    for ip, packet in enumerate(event_packets):
        if packet['packet_type'] != 0:
            continue

        io_group, io_channel, chip, channel = packet['io_group'], packet['io_channel'], packet['chip_id'], packet['channel_id']
        module_id = (io_group-1)//2

        z_offset = this_detector.tpc_offsets[module_id][0]
        x_offset = this_detector.tpc_offsets[module_id][2]
        y_offset = this_detector.tpc_offsets[module_id][1]

        try:
            x, y = this_detector.geometry[(io_group, io_channel, chip, channel)]
        except KeyError as err:
            print("Chip key not found", (io_group, io_channel, chip, channel))
            continue

        hits[0].append(x/10 + x_offset)
        hits[1].append(y/10 + y_offset)
        hits[2].append(this_detector.get_z_coordinate(io_group, io_channel, packet['timestamp']-last_trigger)/10+z_offset)
        hits_index.append('point_%i'%(ip+start_packet))
        hits_text.append('ADC: %i<br>x: %f<br>y: %f<br>t: %i' % (packet['dataword'],
                                                                    x/10,
                                                                    y/10,
                                                                    packet['timestamp']))
        hits_charge.append(packet['dataword'])

    drawn_objects = [go.Scatter3d(x=hits[0],
                                  z=hits[1],
                                  y=hits[2],
                                  mode='markers',
                                  customdata=hits_index,
                                  text=hits_text,
                                  hoverinfo='text',
                                  showlegend=False,
                                  marker={'size': 1.75,
                                          'opacity': 0.7,
                                          'colorscale': 'Spectral_r',
                                          'colorbar':{
                                              'title': 'ADC counts',
                                              'titlefont': {'size':12},
                                              'tickfont': {'size': 10},
                                              'thickness':15,
                                              'len':0.5,
                                              'yanchor':'bottom',
                                            },
                                          'color': hits_charge})]

    return drawn_objects

def plot_tracks(tracks, track_ids, n_events):
    """Plot 3D tracks starting from MC truth information"""
    pdgs = []
    drawn_objects = []
    for itrk in track_ids:
        track = tracks[itrk]
        if track['dx'] < 1:
            continue
        html_name = Particle.from_pdgid(track['pdgId']).html_name.replace('&pi;','&#960;').replace('&gamma;','&#947;')
        line = go.Scatter3d(x=np.linspace(track['x_start'], track['x_end'], 5),
                            y=np.linspace(track['z_start'], track['z_end'], 5),
                            z=np.linspace(track['y_start'], track['y_end'], 5),
                            mode='lines',
                            hoverinfo='text',
                            name=f'{html_name}',
                            text=f'{html_name}<br>trackId: {itrk}',
                            opacity=0.3,
                            legendgroup=f"{n_events}_{track['pdgId']}",
                            # legendgrouptitle_text=r'$%s$' % html_name,
                            customdata=[f'track_{itrk}'],
                            showlegend=track['pdgId'] not in pdgs,
                            line=dict(
                                color=COLOR_DICT[track['pdgId']],
                                width=12
                            ))
        if track['pdgId'] not in pdgs:
            pdgs.append(track['pdgId'])
        drawn_objects.append(line)

    return drawn_objects


class DetectorGeometry():
    """Class describing the geometry of the Detector"""

    def __init__(self, detector_properties, pixel_layout):
        self.detector_properties = detector_properties
        self.pixel_layout = pixel_layout
        self.geometry = {}
        self.io_group_io_channel_to_tile = {}
        self.tile_positions = None
        self.tile_orientations = None
        self.tpc_offsets = None
        self.load_geometry()
        consts.load_properties(detector_properties,pixel_layout)
        self.tile_map = detector.TILE_MAP
        self.module_to_io_groups = detector.MODULE_TO_IO_GROUPS
        self.tpc_borders = detector.TPC_BORDERS
        self.v_drift = detector.V_DRIFT

    @staticmethod
    def rotate_pixel(pixel_pos, tile_orientation):
        return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]

    def load_geometry(self):
        geometry_yaml = yaml.load(open(self.pixel_layout), Loader=yaml.FullLoader)

        pixel_pitch = geometry_yaml['pixel_pitch']
        chip_channel_to_position = geometry_yaml['chip_channel_to_position']
        self.tile_orientations = geometry_yaml['tile_orientations']
        self.tile_positions = geometry_yaml['tile_positions']
        tile_chip_to_io = geometry_yaml['tile_chip_to_io']
        xs = np.array(list(chip_channel_to_position.values()))[:, 0] * pixel_pitch
        ys = np.array(list(chip_channel_to_position.values()))[:, 1] * pixel_pitch
        x_size = max(xs)-min(xs)+pixel_pitch
        y_size = max(ys)-min(ys)+pixel_pitch

        tile_geometry = {}

        with open(self.detector_properties) as df:
            detprop = yaml.load(df, Loader=yaml.FullLoader)

        self.tpc_offsets = detprop['tpc_offsets']
        for module_id in detprop['module_to_io_groups']:
            for tile in tile_chip_to_io:
                tile_orientation = self.tile_orientations[tile]
                tile_geometry[tile] = self.tile_positions[tile], self.tile_orientations[tile]

                for chip in tile_chip_to_io[tile]:
                    io_group_io_channel = tile_chip_to_io[tile][chip]
                    io_group = io_group_io_channel//1000 + (module_id-1)*2
                    io_channel = io_group_io_channel % 1000
                    self.io_group_io_channel_to_tile[(io_group, io_channel)] = tile

                for chip_channel in chip_channel_to_position:
                    chip = chip_channel // 1000
                    channel = chip_channel % 1000

                    try:
                        io_group_io_channel = tile_chip_to_io[tile][chip]
                    except KeyError:
                        continue

                    io_group = io_group_io_channel // 1000 + (module_id-1)*2
                    io_channel = io_group_io_channel % 1000
                    x = chip_channel_to_position[chip_channel][0] * \
                        pixel_pitch - x_size / 2 + pixel_pitch / 2
                    y = chip_channel_to_position[chip_channel][1] * \
                        pixel_pitch - y_size / 2 + pixel_pitch / 2

                    x, y = self.rotate_pixel((x, y), tile_orientation)
                    x += self.tile_positions[tile][2]
                    y += self.tile_positions[tile][1]
                    self.geometry[(io_group, io_channel, chip, channel)] = x, y

    def get_z_coordinate(self, io_group, io_channel, time):
        tile_id = self.get_tile_id(io_group, io_channel)
        z_anode = self.tile_positions[tile_id][0]
        drift_direction = self.tile_orientations[tile_id][0]
        return z_anode + time * self.v_drift * drift_direction

    def get_tile_id(self, io_group, io_channel):
        if (io_group, io_channel) in self.io_group_io_channel_to_tile:
            tile_id = self.io_group_io_channel_to_tile[io_group, io_channel]
        else:
            return np.nan

        return tile_id
