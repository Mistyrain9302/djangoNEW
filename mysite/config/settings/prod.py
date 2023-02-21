from .base import *

ALLOWED_HOSTS = ['15.164.66.24']

STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
    '/home/ubuntu/projects/djangoNEW/mysite/static/',
]

DEBUG = False