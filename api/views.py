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
import matplotlib.patches as patches

MAGIC = 0.1156069364


DATA_FILE = ''
SENSOR_COORDINATES_DATA = 'media/sensor_coordinates.csv'
IGNORE_DATA_BEFORE_ROW = 4871
STRAIN_TO_LOAD_DATA = 'media/strain_load.csv'
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

def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return s
    return -1

def get_sensors_date(DAY):
    # Load the sensor reading data.
        df = pd.read_csv(DATA_FILE, header=None)
        df = df.iloc[IGNORE_DATA_BEFORE_ROW:]
        df.set_index(pd.to_datetime(df[0], unit='ms'), inplace=True)
        df.replace(to_replace=0, method='ffill', inplace=True)
        time_list = list(map(str,list(df.index)))
        str_index = index_containing_substring(time_list,DAY)
        PLOT_TIMESTAMP = str_index.split('.')[0]
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
        return sensors_data

def calculate_heatmap(circles,rectangles,LABELS):
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

    return blurred

def add_circles(circles,ax,LABELS):
    ax.scatter(
    x=[c['x'] for c in circles],
    y=[c['y'] for c in circles],
    c=[c['color'] for c in circles],
    s=[10],
    zorder=2)

    for index, circle in enumerate(circles):
        offset_y = -15 if index % 2 else 15
        label_x = circle['x']
        label_y = (circle['y'] + 6) + offset_y
        if LABELS == 'loading':
            text = circle['Name'] + ": "+'{}lbs'.format(round(circle['loading']))
        elif LABELS == 'capacity_load':
            text = circle['Name'] + ": "+'{}%'.format(round(circle['capacity_load']))

        ax.text(label_x, label_y, text, color='white', fontsize=9, fontweight=1000)
        ax.text(label_x, label_y, text, fontsize=9, fontweight=600)

def add_rectangles(rectangles,ax,LABELS,sensors_data):
    # Add rectangles.
    for rectangle in rectangles:
        sensor_reading = sensors_data
        rectangle_x = rectangle['x']
        rectangle_y = rectangle['y']
        label_x = rectangle['x'] - 5
        label_y = rectangle['y'] + 6
        rect = patches.Rectangle(
            xy=(rectangle_x, rectangle_y),
            width=rectangle['height'],
            height=rectangle['height'],
            edgecolor='none',
            zorder=1,
            facecolor=rectangle['color'])
        ax.add_patch(rect)
        if LABELS == 'loading':
            text = rectangle['Name']+ ": " +'{}lbs'.format(round(rectangle['loading']))
        elif LABELS == 'capacity_load':
            text = rectangle['Name']+ ": " +'{}%'.format(round(rectangle['capacity_load']))
        ax.text(label_x, label_y, text, fontsize=9, fontweight=600)

    
def produce_image(blurred,circles,rectangles,LABELS,sensors_data):
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
    roof_image = imread(settings.MEDIA_ROOT+'/'+'roof_transparent.png')
    add_circles(circles,ax,LABELS)
    add_rectangles(rectangles,ax,LABELS,sensors_data)
    ax.imshow(roof_image, aspect=1)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(IMAGE_FILE,progressive=True,bbox_inches='tight')
    with open(IMAGE_FILE, "rb") as img_file:
        img_data = base64.b64encode(img_file.read())
        
    return img_data

def all_day_list(the_list,substring):
    data_list = []
    for i, s in enumerate(the_list):
        if substring in s:
              data_list.append(s.split('.')[0])
    return data_list


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
        latest_time = latest_time.strftime('%Y-%m-%d')
        
        
        file_serializer = FileSerializer(data=request.data)
        #--------Get the file path in the server--------------#
        global DATA_FILE
        k = request.data['file'].name
        DATA_FILE = settings.MEDIA_ROOT +'/' + k
        
        
        #--------Get the maxim data of the latest day among the sensors---------#
        
        # Load the sensor placement data.
        df_sensors = pd.read_csv(SENSOR_COORDINATES_DATA)
        df_offset = df.head(1)
        df_offset = df_offset.to_dict('split')['data'][0]
        
        #Load strain to load data
        df_load = pd.read_csv(STRAIN_TO_LOAD_DATA,header=None)
        df_load = df_load.T
        df_load.columns = df_load.iloc[0]
        df_newload = df_load[1:]
        weights = pd.to_numeric(df_newload.iloc[:,2],errors='coerce')
        strains = pd.to_numeric(df_newload.iloc[:,1],errors='coerce')
        df_newload['load/strain'] = weights/strains
        df_merge = pd.merge(left=df_sensors, right=df_newload,left_on='Name', right_on='Sensor')
        df_max = df_merge[['ID','Name','load/strain']]
        
        time_list = list(map(str,list(df.index)))
        DAY = latest_time
        data_list = all_day_list(time_list,DAY)
        
        max_sensors_data = {}
        df_day = pd.DataFrame()
        for day in data_list:
                PLOT_TIMESTAMP = day
                df_day = pd.concat([df_day, df[PLOT_TIMESTAMP]], ignore_index=True)
        for i in range(1, len(df_day.columns), 3):
                max_sensors_data[str(df_day.iloc[1,i])] = (df_day.iloc[:,i+1].max() - df_offset[i+1]) * MAGIC
        max_load_at_day = []
        for index, row in df_max.iterrows():
                reading = abs(max_sensors_data[str(row['ID'])])
                max_load = row['load/strain']*reading
                sensor = row['Name']
                max_load_at_day.append(max_load)
        
        max_load = max(max_load_at_day)      
        res_data = []
        res_data.append(latest_time)
        res_data.append(round(max_load))
        
        if file_serializer.is_valid():
            file = file_serializer.save()
            return Response(res_data, status=status.HTTP_200_OK)
        else:
            return Response("Failed to Upload", status=status.HTTP_204_NO_CONTENT)


class CapacityHeatmapView(APIView):
    
    def post(self, request, *args, **kwargs):
        
        DAY = request.data['date']  
        sensors_data = get_sensors_date(DAY)
        df_sensors = pd.read_csv(SENSOR_COORDINATES_DATA)
        circles = []
        rectangles = []
        colors = generate_color_scale()
        for index, row in df_sensors.iterrows():
            if 'C' in row['Name']:
                reading = sensors_data[str(row['ID'])]
                reading = reading * -1 if reading < 0 else reading
                capacity_percent = (reading / row['Capacity'])*100
                color = colors[hardcode_color(row['Name'])]
                
                circle = {
                    'x': row['X']-70,
                    'y': row['Y']-70,
                    'color': color.rgb,
                    'size': row['Width'],
                    'reading': reading,
                    'Name':row['Name'],
                    'capacity_load': capacity_percent}
                circles.append(circle)

            elif 'P' in row['Name']:
                reading = sensors_data[str(row['ID'])]
                reading = reading * -1 if reading < 0 else reading
                capacity_percent = (reading / row['Capacity'])*100
                color = colors[hardcode_color(row['Name'])]
                
                rectangle = {
                    'x': row['X']-70,
                    'y': row['Y']-70,
                    'color': color.rgb,
                    'width': row['Width'],
                    'height': row['Height'],
                    'reading': reading,
                    'Name':row['Name'],
                    'capacity_load': capacity_percent}
                rectangles.append(rectangle)
       
        blurred = calculate_heatmap(circles,rectangles,"capacity_load")
        img_data = produce_image(blurred,circles,rectangles, "capacity_load",sensors_data)
        
        return Response(img_data, status=status.HTTP_200_OK)

class LoadHeatmapView(APIView):
    
     def post(self, request, *args, **kwargs):
        
        DAY = request.data['date']  
        sensors_data = get_sensors_date(DAY)
        df_sensors = pd.read_csv(SENSOR_COORDINATES_DATA)
        circles = []
        rectangles = []
        colors = generate_color_scale()
        
        #--Load data
        df_load = pd.read_csv(STRAIN_TO_LOAD_DATA,header=None)
        df_load_trans = df_load.T
        df_load_trans.columns = df_load_trans.iloc[0]
        df_newload = df_load_trans[1:]
        weights = pd.to_numeric(df_newload.iloc[:,2],errors='coerce')
        strains = pd.to_numeric(df_newload.iloc[:,1],errors='coerce')
        df_newload['load/strain'] = weights/strains
        
        #--Merge two dataframes
        df_merge = pd.merge(left=df_sensors, right=df_newload,left_on='Name', right_on='Sensor')
        
        circles = []
        rectangles = []
        colors = generate_color_scale()
        for index, row in df_merge.iterrows():
            if 'C' in row['Name']:
                reading = sensors_data[str(row['ID'])]
                reading = reading * -1 if reading < 0 else reading
                loading = reading * row['load/strain']
                color = colors[hardcode_color(row['Name'])]
              
                circle = {
                    'x': row['X']-70,
                    'y': row['Y']-70,
                    'color': color.rgb,
                    'size': row['Width'],
                    'reading': reading,
                    'Name':row['Name'],
                    'loading': loading}
                circles.append(circle)

            elif 'P' in row['Name']:
                reading = sensors_data[str(row['ID'])]
                reading = reading * -1 if reading < 0 else reading
                loading = reading * row['load/strain']
                color = colors[hardcode_color(row['Name'])]
               
                rectangle = {
                    'x': row['X']-70,
                    'y': row['Y']-70,
                    'color': color.rgb,
                    'width': row['Width'],
                    'height': row['Height'],
                    'Name':row['Name'],
                    'reading': reading,
                    'loading': loading}
                rectangles.append(rectangle)
        blurred = calculate_heatmap(circles,rectangles,"loading")
        img_data = produce_image(blurred,circles,rectangles,"loading",sensors_data)
        
        return Response(img_data, status=status.HTTP_200_OK)
    
def all_data_list(df,df_offset,df_max,the_list,substring,sensors):
        data_list = []
        for i, s in enumerate(the_list):
            if substring in s:
                  data_list.append(s.split('.')[0])
        max_sensors_data = {}
        df_day = pd.DataFrame()
        for day in data_list:
            PLOT_TIMESTAMP = day
            df_day = pd.concat([df_day, df[PLOT_TIMESTAMP]], ignore_index=True)
        for i in range(1, len(df_day.columns), 3):
                max_sensors_data[str(df_day.iloc[1,i])] = (df_day.iloc[:,i+1].max() - df_offset[i+1]) * MAGIC
        max_load_at_day = []
        for index, row in df_max.iterrows():
            max_sensor_value = {}
            reading = abs(max_sensors_data[str(row['ID'])])
            max_load = round(row['load/strain']*reading)
            sensor = row['Name']
            if sensor in sensors:
                max_sensor_value = {
                    sensor:max_load
                }
                max_load_at_day.append(max_sensor_value)
        return max_load_at_day

def get_max_data(DAYs,sensors,time_list,df,df_offset,df_max):
    data_at_days = []
    
    for DAY in DAYs:
        data_at_day = {}
        data_list = all_data_list(df,df_offset,df_max,time_list,DAY,sensors)
        data_at_day = {
            DAY:data_list
        }
        data_at_days.append(data_at_day)
    return data_at_days
           
class LoadChronologyView(APIView):
    
    def post(self, request, *args, **kwargs):
        # Load the sensor reading data.
        df = pd.read_csv(DATA_FILE, header=None)
        df.set_index(pd.to_datetime(df[0], unit='ms'), inplace=True)
        
        # Load the sensor placement data.
        df_sensors = pd.read_csv(SENSOR_COORDINATES_DATA)
        df_offset = df.head(1)
        df_offset = df_offset.to_dict('split')['data'][0]
        
        #Load strain to load data
        df_load = pd.read_csv(STRAIN_TO_LOAD_DATA,header=None)
        df_load = df_load.T
        df_load.columns = df_load.iloc[0]
        df_newload = df_load[1:]
        weights = pd.to_numeric(df_newload.iloc[:,2],errors='coerce')
        strains = pd.to_numeric(df_newload.iloc[:,1],errors='coerce')
        df_newload['load/strain'] = weights/strains
        df_merge = pd.merge(left=df_sensors, right=df_newload,left_on='Name', right_on='Sensor')
        df_max = df_merge[['ID','Name','load/strain']]
        
        time_list = list(map(str,list(df.index)))
        DAYs = request.data['last_week']
        sensors = request.data['sensors']
        data = get_max_data(DAYs,sensors,time_list,df,df_offset,df_max)
        
        
        updated = []
        for day_data in data:
            renewed_day = {
                'name':list(day_data.keys())[0]
            }
            for item in list(day_data.values())[0]:
                renewed_day.update(item)
            updated.append(renewed_day)
        return Response(updated)