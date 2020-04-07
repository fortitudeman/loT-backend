from django.urls import path
from .views import *

urlpatterns = [
    path('upload/', FileUploadView.as_view()),
    path('capacity-heatmap/',CapacityHeatmapView.as_view()),
    path('load-heatmap/',LoadHeatmapView.as_view()),
    path('chronology/',LoadChronologyView.as_view())
]