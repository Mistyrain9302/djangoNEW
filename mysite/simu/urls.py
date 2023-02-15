from django.urls import path
from simu.dash_apps.finished_apps import simple_example
from . import views

app_name = 'simu'

urlpatterns = [
    path('', views.dist_base, name='dist_base'),
    path('classify/',views.classify,name='classify'),
    path('regress/',views.regress,name='regress'),
]