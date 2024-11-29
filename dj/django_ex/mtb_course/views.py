
from django.views.generic import ListView,DetailView
from .models import MtbCourse

class CourseListView(ListView):
    model = MtbCourse
    template_name='mtb_course/course_list.html'
    context_object_name = 'courses'

class CourseDetailView(DetailView):
    model = MtbCourse
    template_name ='mtb_course/course_detail.html'
    context_object_name ='courses'



























# from django.shortcuts import render
# from .models import MtbCourse, Region

# def course_list(request): 
#     region = request.GET.get('region')
#     difficulty = request.GET.get('difficulty')

#     courses = MtbCourse.objects.all()

#     if region:
#         courses = courses.filter(region_name = region)
    
#     if difficulty:
#         courses = courses.filter(difficulty=difficulty)
    
#     regions = Region.objects.all()

#     return render(request,'courses/course_list.html',{
#         'courses':courses,
#         'regions':regions
#     })

#     def course_detail(request, course_id):
#         course = MtbCourse.objects.get(id=course_id)
#         return(request,'course/course_detail.html',{'course':course})
