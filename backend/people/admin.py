from django.contrib import admin

from .models import FaceTemplate, Person

admin.site.register(Person)
admin.site.register(FaceTemplate)