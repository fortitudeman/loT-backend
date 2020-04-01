from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .serializers import FileSerializer
from rest_framework.decorators import api_view
from .models import File
from django.conf import settings
import numpy as np
import pandas as pd
from colour import Color
from itertools import chain
from matplotlib.image import imread
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import base64


MAGIC = 0.1156069364
LABELS = 'capacity_load'

DATA_FILE = ''
SENSOR_COORDINATES_DATA = 'media/sensor_coordinates.csv'
IGNORE_DATA_BEFORE_ROW = 4871
PLOT_TIMESTAMP = '2020-03-10 14:55:58'
SOURCE_IMAGE_WIDTH = 1000
SOURCE_IMAGE_HEIGHT = 680

def generate_color_scale():
    red = Color('#ff6d6a')
    yellow = Color('#fec359')
    green = Color('#76c175')
    blue = Color('#54a0fe')

    scale = []
   
    scale = red.range_to(blue, 100)

    return list(scale)

def hardcode_color(sensor_name):
    names_map = {
        'C01':0,
        'P01':5,
        'C02':15,
        'P02':15,
        'P05':20,
        'P06':30,
        'P08':40,
        'P07':90,
        'C03':20,
        'C04':20,
        'C05':65,
        'C06':65,
        'C07':70,
        'C08':70,
        'C09':95,
        'P03':85,
        'P04':95}
    return names_map[sensor_name]


class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
      
      #--------Get the latest time from the timestamp--------#  
      file = request.data['file']
      df = pd.read_csv(file, header=None)
      df.set_index(pd.to_datetime(df[0], unit='ms'), inplace=True)
      df.replace(to_replace=0, method='ffill', inplace=True)
      latest_time = list(df.index)[-1]  
      latest_time = latest_time.replace(second=0,microsecond=0)
      latest_time = latest_time.strftime('%Y-%m-%d %H:%M')
      
      
      file_serializer = FileSerializer(data=request.data)
      #--------Get the file path in the server--------------#
      global DATA_FILE
      k = request.data['file'].name
      DATA_FILE = settings.MEDIA_ROOT +'/' + k
      
      if file_serializer.is_valid():
          file = file_serializer.save()

          return Response(latest_time, status=status.HTTP_201_CREATED)
      else:
          return Response("Failed to Upload", status=status.HTTP_400_BAD_REQUEST)


class DrawHeatmapView(APIView):
      
      
      def post(self, request, *args, **kwargs):
          
        # Load the sensor reading data.
        df = pd.read_csv(DATA_FILE, header=None)
        df = df.iloc[IGNORE_DATA_BEFORE_ROW:]
        df.set_index(pd.to_datetime(df[0], unit='ms'), inplace=True)
        df.replace(to_replace=0, method='ffill', inplace=True)

        # Load the sensor placement data.
        df_sensors = pd.read_csv(SENSOR_COORDINATES_DATA)

        # Limit the data to specific timestamp.
        df_slice = df[PLOT_TIMESTAMP]
        df_slice = df_slice.to_dict('split')['data'][0]
        df_offset = df.head(1)
        df_offset = df_offset.to_dict('split')['data'][0]

        sensors_data = {}
        for i in range(1, len(df_slice), 3):
            sensors_data[str(df_slice[i])] = (df_slice[i+1] - df_offset[i+1]) * MAGIC

        circles = []
        rectangles = []
        colors = generate_color_scale()
        for index, row in df_sensors.iterrows():
            if 'C' in row['Name']:
                reading = sensors_data[str(row['ID'])]
                reading = reading * -1 if reading < 0 else reading
                capacity_percent = (reading / row['Capacity'])*100
                color = colors[hardcode_color(row['Name'])]
                # Uncomment this line when using real data.
                #color = colors[round(capacity_percent)]
                circle = {
                    'x': row['X'],
                    'y': row['Y'],
                    'color': color.rgb,
                    'size': row['Width'],
                    'reading': reading,
                    'capacity_load': capacity_percent}
                circles.append(circle)

            elif 'P' in row['Name']:
                reading = sensors_data[str(row['ID'])]
                reading = reading * -1 if reading < 0 else reading
                capacity_percent = (reading / row['Capacity'])*100
                color = colors[hardcode_color(row['Name'])]
                # Uncomment this line when using real data.
                #color = colors[round(capacity_percent)]
                rectangle = {
                    'x': row['X'],
                    'y': row['Y'],
                    'color': color.rgb,
                    'width': row['Width'],
                    'height': row['Height'],
                    'reading': reading,
                    'capacity_load': capacity_percent}
                rectangles.append(rectangle)
        # Calculate heatmap.
        combined_data = [{'x':d['x'], 'y': d['y'], LABELS: d[LABELS]} for d in chain(circles, rectangles)]

        def f(x, y):
            for d in combined_data:
                if abs(d['x']-x) < 80 and abs(d['y']-y) < 80:
                    dist = np.linalg.norm(np.array([x, y]) - np.array([d['x'], d['y']]))
                    if dist < 25:
                        return d[LABELS]
            return 0

        heatmap = np.array([[f(x, y) for x in range(SOURCE_IMAGE_WIDTH)] for y in range(SOURCE_IMAGE_HEIGHT)])
        blurred = gaussian_filter(heatmap, sigma=50, mode='mirror', cval=0)

        # Setup the plot.
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create custom colormap.
        red = Color('#ff6d6a')
        yellow = Color('#fec359')
        green = Color('#76c175')
        blue = Color('#54a0fe')
        colors = [blue.rgb, green.rgb, yellow.rgb, red.rgb]
        cmap_name = 'my_list'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        ax.imshow(blurred, interpolation='hamming', cmap=cm)
        IMAGE_FILE = settings.MEDIA_ROOT+'/' + f'new_heatmap_{LABELS}.png'
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(IMAGE_FILE)
        
        with open(IMAGE_FILE, "rb") as img_file:
            img_data = base64.b64encode(img_file.read())
        
        return Response(img_data, status=status.HTTP_201_CREATED)
