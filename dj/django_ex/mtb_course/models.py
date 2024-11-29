from django.db import models

# class Region(models.Model):
#     name = models.CharField(max_length=100)
#     description = models.TextField(blank=True)

#     def __str__(self):
#         return self.name

# class MtbCourse(models.Model):
#     REGION_CHOICES = [
#         ('서울', '서울'),
#         ('경기', '경기'),
#         ('강원', '강원'),
#         ('충북', '충북'),
#         ('충남', '충남'),
#         ('전북', '전북'),
#         ('전남', '전남'),
#         ('경북', '경북'),
#         ('경남', '경남'),
#         ('제주', '제주'),
#     ]
# name = models.CharField(max_length=200)

# region = models.CharField(
#         max_length=100,
#         choices= REGION_CHOICES,
#         default='서울'
#     )
# difficulty = models.CharField(max_length=울0)
# length = models.FloatField()
# description = models.TextField()
# source_url = models.URLField()
# create_at = models.DateTimeField(auto_now_add=True)

# def __str__(self):
#     return f"{self.name} - {self.region}"

# mtb_course/models.py
from django.db import models

class MtbCourse(models.Model):
    REGION_CHOICES = [
        ('서울', '서울'),
        ('경기', '경기'),
        ('강원', '강원'),
        ('충북', '충북'),
        ('충남', '충남'),
        ('전북', '전북'),
        ('전남', '전남'),
        ('경북', '경북'),
        ('경남', '경남'),
        ('제주', '제주'),
    ]

    name = models.CharField(max_length=200)
    region = models.CharField(
        max_length=100,
        choices=REGION_CHOICES,
        default='서울'
    )
    difficulty = models.CharField(max_length=50)
    length = models.FloatField()
    description = models.TextField()
    source_url = models.URLField()

    def __str__(self):
        return self.name