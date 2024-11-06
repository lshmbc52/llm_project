#star= ("*" *5)

# for i in range(1,26,5):
#     print("*****")

#for i in star:
#   print("*****")

# for i in range(1,6):
#     print("*" *i)

# for i in range(1,6):
#     print(i)


#number = int(input("숫자입력:"))

# for i in range(number,0, -1):

#     print(i)


# for i in range(number):
#     print(i+1,end ="")

#     if i // 10 == 0:
#         print(i , end="")
#     if i % 10 ==9:
#         print("\n")

# import random

# randomNumber = random.sample(range(1,46), 6)
# number = int(input("몇장 사나요?"))


# for i in range(number):
#     randomNumber = random.sample(range(1,46),6)


#     print(randomNumber,end="\n")

# for i in randomNumber:
#     print(i,end="\t")

# for i in range(number):
#     randomNumber = random.sample(range(1,46),6)
#     randomNumber.sort()
#     print(randomNumber)

# print("로또 종료")

# class Rectangle():
    
#     def __init__(self,height,width):
#         self.height = height
#         self.width = width
        

#     def area(self):
#         return self.height * self.width

#     def perimeter(self):
#         return 2 *(self.height * self.width)

# rect = Rectangle(5,10)

# print("Area:", rect.area())
# print("Perimeter:", rect.perimeter())

# class Student:
#     def __init__(self,name):
#         self.name = name
#         self.score = []
    
#     def add_grade(self,score):
#             self.score.append(score)
        
#     def average(self):
#         mean = sum(self.score) / len(self.score)
#         return  mean

# student = Student("Kim")

# student.add_grade(89)
# student.add_grade(50)


# print(f"{student.name}'s student average {student.average():.2f}")

# class Book:
#     def __init__(self,title,author):
#         self.title = title
#         self.author = author

#     def __str__(self):
#         #print(f"{self.title},{self.author}")
#         return f"{self.title},{self.author}"


# class Library:
#     def __init__(self):
#         self.books = []
     


#     def add_book(self,book):
#         self.books.append(book)

#     def show_books(self):
#         for book in self.books:
#             print(book)


# library = Library()

# book1 = Book("소년", "한강")
# book2 = Book("이문열","아들")

# library.add_book(book1)
# library.add_book(book2)

# library.show_books()

        
# i = int(input("숫자 입력:"))

# for i in range(i):
#     if i % 10 == 0:
#         print()
#     else:
#         print(i,end = "\t")

# class Animal():
#     def __init__(self, name, sound):
#         self.name = name
#         self.sound = sound

#     def make_sound(self): 
#         print(f"{self.name} makes {self.sound} sound")

# class Dog(Animal):
#     def __init__(self,name):
#         super().__init__(name,"djdjdj")

# class Cat(Animal):
#     def __init__(self, name):
#         super().__init__(name, "mddmdmm")

# dog = Dog("Buddy")
# dog.make_sound()

# cat = Cat("oooo")
# cat.make_sound()


# class Account:
#     def __init__(self,name,balance):
#         self.name = name
#         self.balance = balance

#     def deposit(self,amount):
#         self.balance += amount
#         return self.balance

#     def withdraw(self,amount):
        
#         if self.balance < amount:
#             print(f"Insufficent funds")
#             return None
#         else:
#             self.balance -= amount
#             return self.balance
        
# class Bank():
#     def __init__(self, name):
       
#         self.accounts = []
    
#     def add_account(self,account):
#         if isinstance(account,Account):
#             self.accounts.append(account)

#     def total_balance(self):
#         return sum( account.balance for account in self.accounts)


# account1= Account("Lee",5000)
# account2 =Account("kim", 789)

# account1.deposit(58988)
# account2.withdraw(9580)




# bank = Bank("name")

# bank.add_account(account1)
# bank.add_account(account2)

# print(bank.total_balance())


# class Math:
#     @staticmethod

#     def add(x,y):
#         return x + y
    
#     @staticmethod

#     def multi(x,y):
#         return x * y

# print(Math.add(7,8))

# print(Math.multi(2,3))

# class Person:
#     population = 0

#     def __init__(self,name):
#         self.name = name
#         Person.population += 3

#     @classmethod
#     def get_population(cls):
#         return cls.population

#     @classmethod
#     def create_anonymous(cls):
#         return cls("Anonymous")


# john = Person("john")
# anonymous = Person.create_anonymous()

# print(Person.get_population())


# import random
# import time

# lunch = ["된장찌개","피자","제육볶음","짜장면"]

# while True:
#     print(lunch)
    
#     item = input("음식을 추가:")

#     if item == "q":
#         break
#     else:
#         lunch.append(item)

# print(lunch)

# set_lunch = set(lunch)

# while True:
#     print(set_lunch)
#     item = input("음식을 삭제:")

#     if item == 'q':
#         break
#     else:
#         set_lunch = set_lunch - set([item])

# print(set_lunch, "중에서 선택함")

# print("5")
# time.sleep(1)
# print("4")
# time.sleep(1)
# print("3")
# time.sleep(1)
# print("2")
# time.sleep(1)
# print("1")
# time.sleep(1)

# print(random.choice(list(set_lunch)))


# set_lunch = set(lunch)
# item ="짜장면"

# print(set_lunch - set([item]))

# set_lunch = set_lunch -set([item])
# print(set_lunch)

# total_dict = {}

# while True:
#     question= input("질문 입력해요:")
#     if question == "q":
#         break
#     else:
#         total_dict[question] = ""

# for i in total_dict:
#     print(i)
#     answer = input("답변입력해 주세요:")
#     total_dict[i] = answer



# print(total_dict)

total_list = []

while True:
    question= input("질문 입력해요:")
    if question == "q":
        break
    else:
        total_list.append({"질문": question, "답변": ""})

for i in total_list:
    print(i["질문"])
    
    answer =input("답변 입력해 주오:")
    i["답변"]= answer
print(total_list)




