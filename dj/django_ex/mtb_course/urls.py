from django.urls import path
from . import views

app_name ='mtb_course'

urlpatterns = [
    path('', views.CourseListView.as_view(),name='course_list'),
    path('<int:pk>/', views.CourseDetailView.as_view(), name='course_detail'),
]

