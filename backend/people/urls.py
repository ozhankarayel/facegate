from django.urls import path

from .views import PersonDetailView, PersonListCreateView

urlpatterns = [
    path("people/", PersonListCreateView.as_view()),
    path("people/<int:pk>/", PersonDetailView.as_view()),
]