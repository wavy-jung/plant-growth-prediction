# Plant Growth Prediction Competition




## Competition Info
Background : 한 쌍의 이미지를 입력 값으로 받아 작물의 생육 기간 예측 모델 개발을 목표로 합니다.  
Purpose : 식물 생육을 위한 최적의 환경 도출  
Host : DACON, KIST 강릉  
Page Link : https://dacon.io/competitions/official/235851/overview/description




## Results
Rank : 10 / 97 (Single Competitor)  
Private MAELoss : 5.38558  
<img width="1188" alt="Screen Shot 2021-12-22 at 1 08 54 AM" src="https://user-images.githubusercontent.com/80743307/146962249-1a943c5a-e3fb-4151-bc78-e09e5f18110e.png">




## Modeling
한 쌍의 이미지를 각각 다른 모델 입력으로 넣어 임베딩을 각각 생성하고 이를 이어 붙여 linear 연산을 수행하는 방식  
<img width="500" alt="Screen Shot 2021-12-22 at 1 22 56 AM" src="https://user-images.githubusercontent.com/80743307/146963914-317c06db-8f4b-4130-a7a6-794befe2382e.png">




