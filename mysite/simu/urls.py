from django.urls import path
from simu.dash_apps.finished_apps import simple_example, dist, cluster, regressor
from . import views

app_name = 'simu'

urlpatterns = [
    path('', views.dist_base, name='dist_base'),
    path('cluster',views.cluster, name='cluster'),
    path('regressor',views.regressor, name='regressor'),
    path('scores/',views.scores, name='scores'),
    path('classify/',views.classify,name='classify'),
    path('regress/',views.regress,name='regress'),
]