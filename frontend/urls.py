from django.urls import path
from .views import CriminalDetector

urlpatterns = [
    path('', CriminalDetector.as_view(), name='criminal_detector'),
    # path('upload_image/',  upload_image, name='upload_image'),
]
