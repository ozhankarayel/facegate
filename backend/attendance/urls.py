from django.urls import path

from .views import AccessLogListView, RecognizeView, VerifyView

urlpatterns = [
    path("recognize/", RecognizeView.as_view()),
    path("verify/", VerifyView.as_view()),
    path("logs/", AccessLogListView.as_view()),
]