import numpy as np

def iuf(traindata):
	movieusers = np.zeros(shape=(1000))
	for train in traindata:
		for movid in range(1000):
			if(train[movid] != 0):
				movieusers[movid] += 1
	
	iufv = np.zeros(shape=(1000))
	for m in range(1000):
		if movieusers[m] != 0:
			iufv[m] = np.log(200/movieusers[m])

	newtraindata = np.zeros(shape=(200,1000))
	
	uid = 0
	for train in traindata:
		for i in range(1000):
			if train[i] != 0:
				newtraindata[uid][i] = train[i]*iufv[i]
		uid += 1

	return newtraindata, iufv