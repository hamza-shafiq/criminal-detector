import os

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.views.generic import TemplateView
from django.shortcuts import render
from . import criminal_detector


class CriminalDetector(TemplateView):
    template_name = 'detector.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, context={})

    def post(self, request, *args, **kwargs):
        image = request.FILES.get('image_field', '')
        path = default_storage.save("ML/test/" + image.name, ContentFile(image.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)

        results = criminal_detector.predict_criminal(image_path=tmp_file)
        return render(request, self.template_name, context={"results": results, "img": "media/"+tmp_file})
