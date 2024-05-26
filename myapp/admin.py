# admin.py dosyasında
from django.contrib import admin
from .models import ExamResult

admin.site.register(ExamResult)
