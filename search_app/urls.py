from django.urls import path
from . import views

urlpatterns = [
    path('api/retrieve-serp', views.retrieve_serp)
]