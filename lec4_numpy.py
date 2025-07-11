#넘파이 첫번쨰

import numpy as np
import pandas as pd

a = np.array([1, 2, 3, 4, 5])
b = np.array(["apple", "banana", "orange"])
c = np.array([True, False, True, True])
d = np.array(["q", 2,])
#d = np.array(['q', 2, [1,2,3]])
type(a)
list(a[2:4])
a + 3
a * 20

b = np.array([6,7,8,9,10])
b = a + 5

a + b
#대응되는 숫자끼리 더해짐.

a + b + 8
#대응되는 솟자끼리 더해지고 모든 원소에 8씩 더해졌네

a.cumsum()

np.arange([start, ]stop, [step, ]dtype=None)

np.arange(4, 10, step=0.5)
a = np.arange(4, 10, step=0.5)
len(a)

#1000이하의 7의 배수 발생시키기
vec_a = np.arange(7, 1001, step = 7)
vec_a

#vec_a의 합
sum(vec_a)
vec_a.sum()

#vec_a의 누적합
np.cumsum(vec_a)
vec_a.cumsum()


# pip install palmerpenguins
# 데이터로드
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()

penguins = penguins.dropna()
penguins

#스페너 모양은 속성(매서드 아님!)
penguins.shape

vec_m = np.array(penguins["body_mass_g"])
vec_m.shape

#펭귄의 몸무게 최대.최소
np.max(vec_m)
np.min(vec_m)
np.max(vec_m) + np.min(vec_m)


#펭귄 몸무게의 최대값(번쩨?) 찾기
vec_m.argmax()

#펭귄 몸무게의 평균
vec_m.mean()

#평균이 4.2kg라고 하면, 평균보다 작은 펭귄들은 몇 마리?
#너무 어렵구ㅜㅠㅜ
sum(vec_m < 4200)

#3KG이상인 펭귄들은 몇 마리?
sum(vec_m >= 3000)
sum((vec_m < 4200) & (vec_m >= 3000))

#연습문제 2
#리스트를 튜플로 변환하고, 다시 집합으로 변환한 후
#개별 요수의 개수를 출력하시오. 
nums = [5, 10, 10, 15]
#->우선 기본 값 (nums에 5,10,15,10이 들어있습니다.)
nums_tuple = tuple(nums)
print(nums_tuple)
#->nums를 튜플로 바꾸기
nums_set = set(nums_tuple)
print(nums_set)
#->nums_tuple을 집합으로 바꾸기
print('요소 개수:', len(nums_set))
print(len(nums_set))
#-> 마지막으로 nums_set의 개수는 몇 개인지



#연습문제 3
# 다음 딕셔너리에서 age를 28로 수정하고 city 키-값 쌍을 삭제한 후
# 딕셔너리를 출력하세요.
profile = {
    "name": "Jane",
    "age": 27,
    "city": "Busan"
}
#프로필은 무조건 한번에 돌려야하나?
profile["age"] = 28

del profile["city"]
print("수정된 딕셔너리:", profile)

#연습문제 4
#다음 두 집합의 합집합, 교집합, 차집합을 각각 출력
set_x = {1, 2, 3, 4}
set_y = {3, 4, 5, 6}
#합집합
print("합집합:", set_x.union(set_y))
#교집합
print("교집합:", set_x.intersection(set_y))
#차집합
print("차집합:", set_x.difference(set_y))



#넘파이 행렬 연산 연습
#1
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

import numpy as np

#axis?
np.repeat(8, 4)
arr = np.array([[1, 2], [3, 4]])
np.repeat(arr, 3, axis=1)
np.tile([1, 3, 5], 4)

np.repeat(arr, 3, axis=0).shape
np.repeat(arr, 3, axis=1).shape

import numpy as np

arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])


print(arr)

np.sum(arr)
np.sum(arr, axis=0)
np.mean(arr, axis=10
        
#벡터의길이, 모양, 사이즈
vec_a = np.array([2, 1, 4])
len(vec_a)
vec_a.shape
vec_a.size
len(arr)

# 행렬의 길이, 모양, 사이즈
arr.shape

#브로드캐스팅
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a +b)

#2차원 배열 생성
matrix = np.array([ 0.0, 0.0, 0.0]
                  

# 인덱스
vec_a = np.arange(20)
vec_a

vec_a[:7:2]

a = np.array([1, 5, 7, 8, 10])
a < 7
np.where(a < 7)[0][1]

import numpy as np

vec = np.array([1, 2])                
mat = np.array([[1, 2]])

#행렬
matrix = np.colum_stack(
    (np.arange(1,5),
     np.arange(12, 16)

# 행렬의 구분
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
print(b)

import numpy as np

B = np.array([[5, 6],
              [7, 8]])

print(B)

# 두번째열(기말고사)점수가 10점 이하인 학생들의 데이터 필터링

x[x[:,1] <= 10, :]

np.random.seed(2025)
np.random.choice(np.arange(1, 101),
                 size = 3000, replace = True)
mat_a = vec_a.reshape(-1, 2)
mat_a 


# 여기서부터 다시 공부

# 1. 중간고사 성적이 50이상인 학생들의 데이터를 걸러내기

mat_a([x:[:] 50 <=, :])

import numpy as np

B = np.array([[5, 6],
              [7, 8]])

scores = [45, 67, 50, 38, 90, 52, 49, 73, 31]
passed = [score for score in scores if score >= 50]
print("50점 이상인 학생 수:", len(passed))

import random
random.seed(2025)

# 무작위로 10명의 중간고사 점수 생성 (0~100 사이)
scores = [random.randint(0, 100) for _ in range(10)]

# 50점 이상인 학생만 필터링
passed = [score for score in scores if score >= 50]

# 결과 출력
print("전체 점수:", scores)
print("50점 이상인 학생 수:", len(passed))
print("50점 이상 점수:", passed)

# 2. 중간고사 성적 50이상 학생들 몇명?
import random

# 시드 고정
random.seed(2025)

# 10명의 학생이 있고, 각자 두 개의 중간고사 성적을 가짐
# 예: [(첫 번째 점수, 두 번째 점수), ...]
scores = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]

# 두 번째 점수가 50점 이상인 학생만 필터링
passed = [score for score in scores if score[1] >= 50]

# 결과 출력
print("전체 성적 (1차, 2차):", scores)
print("2차 성적이 50점 이상인 학생 수:", len(passed))
print("2차 성적 50점 이상 학생 점수:", passed)

# 3. 그 학생들의 기말고사 성적 평균은 몇명?

# 4. 중간고사 최고점을 맞은 학생의 기말 성적은?



import random
random.seed(2)
dice = random.randint(1, 6)
print("주사위 눈:", dice)

import random
dice = random.randint(1,6)
print("주사위 눈 :", dice)

import random
game = random.randint(가위, 바위, 보)

# 중간고사 평균과 기말고사 평균
import random
random.seed(2025)

mid_avg = mat_a[:,0].mean()
fin_avg = mat_a[:,1].mean()

mat_a.mean(axis=0)
mid_avg = result[0]
fin_avg = result[1]

#중간고사 성적이 50이상인 학생
mat_a[mat_a[:,0] >=50,:] 
#755

#이 학생들의 기말고사 성적 평균은
mat_a[mat_a[:,0] >=50,:].mean()

#중간고사 최고점을 맞은 기말고사 성적은
mid_score=mat_a[:,0]
fin_score=mat_a[:,1]
mid_a[mid_score] == max(mid_score),1]
#->100 ->17명

#중간고사 최고점을 맞은 학생의 기말고사 성적
mat_a[mid_score > mid_avg,1].mean()
# 51.06891891891892

#중간고사 대비 기말고사 성적이 향상된 학생들 수
sum(mid_score < fin_score)

# 반대로 성적이 떨어진 학생들은 어디에 위치?
# 자 생각해보자, 내가 선생님이야. 점수 떨어진 학생들을
# 내가 상담해줘야지. 근데 학생이 너무 많잖아.
std_index=np.where(mid_score > fin_score)[0]
std_index
mat_a[std_index,:]

#이거 어떻게 하는데...?