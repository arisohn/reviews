eCommerceGAN : A Generative Adversarial Network for E-commerce [paper](https://arxiv.org/abs/1801.03244)
```
@article{DBLP:journals/corr/abs-1801-03244,
  author    = {Ashutosh Kumar and
               Arijit Biswas and
               Subhajit Sanyal},
  title     = {eCommerceGAN : {A} Generative Adversarial Network for E-commerce},
  journal   = {CoRR},
  volume    = {abs/1801.03244},
  year      = {2018},
  url       = {http://arxiv.org/abs/1801.03244},
  archivePrefix = {arXiv},
  eprint    = {1801.03244},
  timestamp = {Thu, 01 Feb 2018 19:52:26 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1801-03244},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# E-commerce GAN
1. 한 줄 요약: 주문 데이터(물품명, 설명, 구매자, 가격, 일자 등)을 저차원의 벡터로 변환하여 GAN을 통하여 그럴듯 하다고 생각되는 주문 데이터를 생성해 낼 수 있다는 것을 적당한 통계적인 수치와 비교를 통하여 보인다.
	1. e-commerce 의 주문을 밀집된 저차원의 벡터(dense and low-dimensional vector)로 변환
	1. ecommerceGAN(ecGAN) 학습: GAN을 통하여 그럴듯한 주문을 생성해 낼 수 있다.
	1. ecommerce-conditional-GAN(ec2GAN) 학습: 특정 조건(상품에 대한)을 가지는 주문을 생성해 낼 수 있다.

## GAN 기초지식
1. Genrative Adversarial Network 설명
1. Deep Convolutional GAN(DCGAN) 설명
1. Wassertein GAN(WGAN) 설명

## 주문 데이터 표현
1. 주문 데이터: 주문은 {customer, product, price, date}로 구성되어 있고 각각을 다른 방법으로 아래와 같이 벡터화 한다.
	1. product embedding: 모든 단어를 word2vec을 통하여 벡터로 변환한 뒤에 이들을 가중 평균(weighted average)을 낸다.
		1. 임의로 선택한 1.43억 물품들의 제목과 물품 설명으로부터 word2vec 모델을 학습한다.
		1. 같은 corpus로부터 각 단어의 IDF(inverse document frequency) 값을 구한다.
		1. sum(word2vec(word) * IDF(word) for all word in title)
		1. 결과 벡터는 128 차원이고 각 스칼라 값은 -1~1 사이의 값을 가진다.
	1. customer embedding: 각각의 고객을 벡터에 embedding하기 위하여, Discriminative Multi-task Recurrent Neural Network(RNN)을 학습시킨다. 
		1. [figure1 추가]
		1. 학습할 목표(multi-class classification task)
			1. 구매할 다음 물품 그룹명(clothes, food, furniture, baby-products etc.)
			1. 구매 가격
			1. 다음 구매까지의 경과일
		1. RNN with LSTM cells
		1. input layer: the sequence of products(product embedding을 통하여 얻은 벡터값)
		1. hidden layer: hidden representation(customer embedding으로 봄). 비슷한 물품 구매 시계열을 갖는 고객이 비슷한 값에 embedding 시킴
		1. 학습: 각 iteration에서 임의의 task(물품 그룹, 구매 가격, 다음 구매 경과일)를 선택하여 학습시킴
		1. 결과 벡터는 128 차원이고 각 스칼라 값은 -1~1 사이의 값을 가진다.
	1. price: 큰 가격의 효과를 줄이기 위하여 log변환을 취하여 -1~1 사이의 값으로 변환한다.
	1. date of purchase: 7차원 벡터로 표현한다.
		1. The first component captures the difference between the current date and a pre-decided epoch
		1. 다음 2개의 컴포넌트는 day of the month
		1. 다음 2개의 컴포넌트는 월을 표현한다.
		1. 일자와 월의 circularity을 확보하기 위하여 unit circle(https://en.wikipedia.org/wiki/Unit_circle)위에 값을 표현 x = (sin(t), cos(t))
		1. 각각의 feature들은 -1 ~ 1로 정규화한다.
	1. 최종 벡터는 128 + 128 + 1 + 7 = 264 차원 벡터로 표현
		1. 128 차원 고객 벡터
		1. 128 차원 제품 벡터
		1. 1차원 가격벡터
		1. 7차원 구매 날짜 벡터

## ecGAN
1. an order O : {Ci, Pj, pk, Dl}
	- Ci
	- Pj
	- pk
	- Dl
1. discriminator: 실제 order와 generator가 생성해내는 order를 구분해내도록 학습한다.
	1. 구조: fully connected layer with two hidden layer, 각 layer의 마지막에는 ReLU activation함수 추가.
1. generator: distriminator가 실제 order와 구분해내짐 못하도록 가짜 order를 생성해낸다.
	1. 구조: fully connected layer with two hidden layer, 각 layer의 마지막에는 ReLU activation함수 추가.
	1. maps the noise vectors(z) to feasible orders(On~)

## ec2GAN
1. [figure 2]
1. generator에 입력으로 들어가는 noise vector(z)에 제품 벡터(Pj)가 추가됨. z' = [z, Pj] 이를 통하여 테스트할 때 원하는 제품에 대한 실현가능한 주문(feasible order)을 생성해 낼 수 있다.
1. generation loss에 reconstruction loss J(R)를 추가했다.
	1. J(R) = || Pj - Pj~ ||, generator에 입력으로 들어가는 제품 벡터와 생성된 제품 벡터의 Euclidean distance
	1. ec2GAN의 loss는 aJ(G)W + (1 - a)J(R), a는 튜닝 파라미터

## ec2GAN의 활용
