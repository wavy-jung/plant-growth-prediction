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


## Experiments Management
WandB Link : https://wandb.ai/doohae/plant-growth?workspace=user-doohae


<img width="1000" alt="image" src="https://user-images.githubusercontent.com/80743307/146965369-7b87e51b-5fa4-4272-8843-c4ea9185868e.png">


## What was good?
- 초기에 wandb 세팅을 하여 실험을 관리하기 매우 편하였음
- 베이스라인을 수정하기 쉽도록 모듈화하여 기능별 코드 관리가 용이하였음


## What to improve?
- 다른 프로젝트와의 병행으로 시간을 많이 투자할 수 없어서 여러 대조실험을 진행하지 못하였음
- 이전 프로젝트에서 모델의 performance 향상을 많이 이끌어냈던 data augmentation 작업을 기초적인 수준만 적용하였음
- 모델링에서 두 모델을 이어주는 CompareNet으로 RNN, CNN 등을 구상하였으나 적용하지 못하였음
