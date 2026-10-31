from django.urls import path

from .views import AccessLogListView, RecognizeView

urlpatterns = [
    path("recognize/", RecognizeView.as_view()),
    path("logs/", AccessLogListView.as_view()),
]