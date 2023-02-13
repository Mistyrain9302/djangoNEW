from django.urls import path

from . import views

app_name = 'simu'

urlpatterns = [
    path('', views.index, name='index'),
]