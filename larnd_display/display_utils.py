import numpy as np
import yaml

from particle import Particle

from larndsim import consts
from larndsim.consts import detector

from collections import defaultdict

import plotly.graph_objects as go


color_dict = {11: '#3f90da',
              -11: '#92dadd',
              13: '#b2df8a',
              -13: '#33a02c',
              22: '#b15928',
              2212: '#bd1f01',
              -2212: '#e76300',
              -211: '#cab2d6',
              211: '#6a3d9a',
              2112: '#555555',
              1000010020: 'blue'}

color_dict = defaultdict(lambda: 'gray', color_dict)

def plot_geometry():
    drawn_objects = []

    for ix in range(0,detector.TPC_BORDERS.shape[0],2):
        for i in range(2):
            for j in range(2):
                drawn_objects.append(go.Scatter3d(x=(detector.TPC_BORDERS[ix][0][j],detector.TPC_BORDERS[ix][0][j]),
                                                  y=(detector.TPC_BORDERS[ix][2][0],detector.TPC_BORDERS[ix+1][2][0]),
                                                  z=(detector.TPC_BORDERS[ix][1][i],detector.TPC_BORDERS[ix][1][i]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))

                drawn_objects.append(go.Scatter3d(x=(detector.TPC_BORDERS[ix][0][j],detector.TPC_BORDERS[ix][0][j]),
                                                  y=(detector.TPC_BORDERS[ix+i][2][0],detector.TPC_BORDERS[ix+i][2][0]),
                                                  z=(detector.TPC_BORDERS[ix][1][0],detector.TPC_BORDERS[ix][1][1]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))

                drawn_objects.append(go.Scatter3d(x=(detector.TPC_BORDERS[ix][0][0],detector.TPC_BORDERS[ix][0][1]),
                                                  y=(detector.TPC_BORDERS[ix+i][2][0],detector.TPC_BORDERS[ix+i][2][0]),
                                                  z=(detector.TPC_BORDERS[ix][1][j],detector.TPC_BORDERS[ix][1][j]),
                                                  showlegend=False,
                                                  hoverinfo='skip',
                                                  opacity=0.5,
                                                  mode='lines',line={'color':'gray'}))

        xx = np.linspace(detector.TPC_BORDERS[ix][0][0], detector.TPC_BORDERS[ix][0][1], 2)
        zz = np.linspace(detector.TPC_BORDERS[ix][1][0], detector.TPC_BORDERS[ix][1][1], 2)
        xx,zz = np.meshgrid(xx,zz)

        single_color=[[0.0, 'rgb(200,200,200)'], [1.0, 'rgb(200,200,200)']]
        z_cathode = (detector.TPC_BORDERS[ix][2][0]+detector.TPC_BORDERS[ix+1][2][0])/2

        cathode_plane=dict(type='surface', x=xx, y=np.full(xx.shape, z_cathode), z=zz,
                           opacity=0.15,
                           hoverinfo='skip',
                           text='Cathode',
                           colorscale=single_color,
                           showlegend=False,
                           showscale=False)

        drawn_objects.append(cathode_plane)

        annotations_x = [(p[0][0]+p[0][1])/2 for p in detector.TPC_BORDERS]
        annotations_y = [p[1][1]for p in detector.TPC_BORDERS]
        annotations_z = [(p[2][0]+p[2][1])/2 for p in detector.TPC_BORDERS]

        annotations_label = ["(%i,%i)" % (ip//2+1,ip%2+1) for ip in range(detector.TPC_BORDERS.shape[0])]
        module_annotations = go.Scatter3d(
            mode='text',
            x=annotations_x,
            z=annotations_y,
            y=annotations_z,
            text=annotations_label,
            opacity=0.5,
            textfont=dict(
                color='gray',
                size=8
            ),
            showlegend=False,
        )
        drawn_objects.append(module_annotations)

    return drawn_objects


def plot_hits(geometry, event_packets, start_packet, last_trigger):
    hits = [[], [], []]
    hits_index = []
    hits_charge = []
    hits_text = []

    for ip, packet in enumerate(event_packets):
        if packet['packet_type'] != 0:
            continue

        io_group, io_channel, chip, channel = packet['io_group'], packet['io_channel'], packet['chip_id'], packet['channel_id']
        module_id = (io_group-1)//4
        z_offset = geometry.tpc_offsets[module_id][0]
        x_offset = geometry.tpc_offsets[module_id][2]
        y_offset = geometry.tpc_offsets[module_id][1]

        io_group = io_group - (io_group-1)//4*4
        x,y = geometry.geometry[(io_group, io_channel, chip, channel)]
        hits[0].append(x/10 + x_offset)
        hits[1].append(y/10 + y_offset)
        hits[2].append(geometry.get_z_coordinate(io_group, io_channel, packet['timestamp']-last_trigger)/10+z_offset)
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
                                          'opacity': 0.5,
                                          'colorscale': 'Spectral_r',
                                          'color': hits_charge})]

    return drawn_objects

def plot_tracks(tracks, track_ids, n_events):
    pdgs = []
    drawn_objects = []
    for itrk in track_ids:
        track = tracks[itrk]
        latex_name = Particle.from_pdgid(track['pdgId']).latex_name
        html_name = Particle.from_pdgid(track['pdgId']).html_name.replace('&pi;','&#960;').replace('&gamma;','&#947;')
        line = go.Scatter3d(x=np.linspace(track['x_start'], track['x_end'], 5),
                            y=np.linspace(track['z_start'], track['z_end'], 5),
                            z=np.linspace(track['y_start'], track['y_end'], 5),
                            mode='lines',
                            hoverinfo='text',
                            name=r'$%s$' % latex_name,
                            text='%s<br>trackId: %i' % (html_name, itrk),
                            opacity=0.5,
                            legendgroup='%i_%i' % (n_events,track['pdgId']),
                            # legendgrouptitle_text=r'$%s$' % latex_name,
                            customdata=['track_%i' % itrk],
                            showlegend=track['pdgId'] not in pdgs,
                            line=dict(
                                color=color_dict[track['pdgId']],
                                width=12
                            ))
        if track['pdgId'] not in pdgs:
            pdgs.append(track['pdgId'])
        drawn_objects.append(line)

    return drawn_objects


class DetectorGeometry():
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
        for tile in tile_chip_to_io:
            tile_orientation = self.tile_orientations[tile]
            tile_geometry[tile] = self.tile_positions[tile], self.tile_orientations[tile]

            for chip in tile_chip_to_io[tile]:
                io_group_io_channel = tile_chip_to_io[tile][chip]
                io_group = io_group_io_channel//1000
                io_channel = io_group_io_channel % 1000
                self.io_group_io_channel_to_tile[(io_group, io_channel)] = tile

            for chip_channel in chip_channel_to_position:
                chip = chip_channel // 1000
                channel = chip_channel % 1000

                try:
                    io_group_io_channel = tile_chip_to_io[tile][chip]
                except KeyError:
                    continue

                io_group = io_group_io_channel // 1000
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
        return z_anode + time * detector.V_DRIFT * drift_direction

    def get_tile_id(self, io_group, io_channel):
        if (io_group, io_channel) in self.io_group_io_channel_to_tile:
            tile_id = self.io_group_io_channel_to_tile[io_group, io_channel]
        else:
            return np.nan

        return tile_id

