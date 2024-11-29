# mtb_course/admin.py
from django.contrib import admin
from .models import MtbCourse

@admin.register(MtbCourse)
class MtbCourseAdmin(admin.ModelAdmin):
    list_display = ['name', 'region', 'difficulty', 'length']  # 관리자 목록에서 표시할 필드
    list_filter = ['region', 'difficulty']  # 필터 옵션
    search_fields = ['name', 'description']  # 검색 가능한 필드

fields = ['name', 'region', 'difficulty', 'length', 'description', 'source_url']