import requests
from bs4 import BeautifulSoup
from .models import MtbCourse, Region


def crawl_mtb_courses():

    base_url = 'https://bikelife.cycling.or.kr/team/info/'
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    courses= soup.find_all('div',class_ = 'course-item')

    for course in courses:
        name = course.find('h2').text
        region_name = course.find('span',class_= 'region').text
        difficulty = course.find('span',class_ = 'difficulty').text

        region,created = Region.objects.get_or_create(name=region_name)

        MtbCourse.objects.create(
            name = name,
            region = region,
            difficulty = difficulty,
        )