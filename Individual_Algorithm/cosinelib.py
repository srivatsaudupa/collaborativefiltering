import csv
import numpy as np
import math
import sys
from iuf import iuf

class Cosine:

	def __init__(self, testfile, K):
		self.K = K
		self.TESTFILE = testfile
		if self.TESTFILE == "test5.txt":
			self.UIDOFF = 201
			self.RESULTFILE = "result5.txt"
			self.MOVIECOUNT = 5	
		elif self.TESTFILE == "test10.txt":
			self.UIDOFF = 301
			self.RESULTFILE = "result10.txt"	
			self.MOVIECOUNT = 10
		else:
			self.UIDOFF = 401
			self.RESULTFILE = "result20.txt"			
			self.MOVIECOUNT = 20
		self.TRAINFILE = "train.txt"
	
	def compute_cosine_similarity(self, testvector, trainvector, caseamp = False):
		
		testhasrated = testvector!=0
		trainhasrated = trainvector!=0
		movielist = np.array(list(range(1000)))
		commonrated = movielist[[np.bitwise_and(testhasrated,trainhasrated)]]
		if len(commonrated) == 0:
			return 0

		dotprod, trainsqr, testsqr = 0,0,0
		for mov in commonrated:
			dotprod += testvector[mov]*trainvector[mov]
			trainsqr += trainvector[mov]**2
			testsqr += testvector[mov]**2
		cosim = dotprod / (np.sqrt(trainsqr)*np.sqrt(testsqr))
		if caseamp:
			cosim *= cosim**1.5
		return cosim

	def get_k_nearest(self, k, ksimmatrix):
		topkmat = {}
		
		for ksim in ksimmatrix:
			appendable = [ksim[1], ksim[2]]
			if(ksim[0] in topkmat):
				topkmat[ksim[0]].append(appendable)
			else:
				topkmat[ksim[0]] = []
				topkmat[ksim[0]].append(appendable)
		
		for key, vals in topkmat.items():
			newvals = []
			vals.sort(key = lambda x:x[1], reverse=True)
			del topkmat[key]
			topkmat[key] = vals
		return topkmat

	def compute(self, caseamp=False, isiuf=False):
		trainfile = csv.reader(open(self.TRAINFILE, "r"), delimiter='\t')
		train_data_list = list(trainfile)
		traindata = np.array(train_data_list).astype(int)

		testfile = csv.reader(open(self.TESTFILE, 'r'), delimiter=' ')
		test_data_list = list(testfile)
		testdata = np.array(test_data_list).astype(int)

		test_userid = np.unique(testdata[:, 0])
		test_userid_count = len(test_userid)

		test_user_array = np.zeros(shape=(test_userid_count, 1000))
		testsum = np.zeros(shape=(100))
		for test in testdata:
			test_user_array[test[0]-self.UIDOFF][test[1]-1] = test[2]
			testsum[test[0]-self.UIDOFF] += test[2]

		if isiuf:
			iuftraindata, iufv = iuf(traindata)
		else:
			iuftraindata = traindata

		cosinsimmat = np.zeros(shape=(100, 200))
		test_uid = 0
		for testuser in test_user_array:
			train_uid = 0
			for trainuser in iuftraindata:
				cosinsimmat[test_uid][train_uid] = self.compute_cosine_similarity(testuser, trainuser, caseamp)
				train_uid += 1
			test_uid += 1

		csv.writer(open("cosinsim"+str(self.MOVIECOUNT)+".csv", "w", newline='')).writerows(list(cosinsimmat))

		maxcosine = np.zeros(shape=(100, 2))
		i= 0

		truid = 0
		tuser = np.zeros(shape=(200))
		tuserap = []
		for train in traindata:
			rated = [trate in train for trate in train if trate != 0]
			totalrated = len(rated)
			tuser[truid] = totalrated
			tuserap.append([truid, totalrated])
			truid += 1

		csv.writer(open("user"+str(self.MOVIECOUNT)+".csv", "w", newline='')).writerows(list(tuserap))

		ksimmatrix = []
		for cosim in cosinsimmat:
			j = 0
			for cosval in cosim:
				ksimmatrix.append([i+self.UIDOFF, j+1, cosval*tuser[j]])
				j+=1
			i+=1

		topkmat = self.get_k_nearest(self.K, ksimmatrix)
		torate = []

		for test in testdata:
			if(test[2] == 0):
				torate.append(test)

		testrate = []
		considerusers = []
		print(self.K)
		for rate in torate:
			userid = rate[0]
			movid = rate[1]-1
			simkusers = topkmat[userid]
			wCsinSum, wCsinProd, k = 0, 0, 0
			for simusr in simkusers:
				movrat = traindata[simusr[0]-1][movid]

				if(movrat == 0):
					continue
				wCsinProd += movrat * simusr[1] 
				wCsinSum += simusr[1]
				k += 1
				considerusers.append([userid, movid+1, simusr[0], simusr[1]])
				if(k >= self.K):
					break;
			if(wCsinSum != 0):
				movrat = int(round(wCsinProd / wCsinSum))
			else:
				movrat = int(round(testsum[userid-self.UIDOFF]*1.0 / self.MOVIECOUNT))

			if movrat > 5:
				movrat = 5
			elif movrat < 1:
				movrat = 1

			testrateval = [userid, movid+1, movrat]
			testrate.append(testrateval)

		testratefile = csv.writer(open("item_"+self.RESULTFILE, "w", newline=''), delimiter=' ')
		testratelist = list(testrate)
		testratefile.writerows(testratelist)

		consratefile = csv.writer(open("cons"+str(self.MOVIECOUNT)+".txt", "w", newline=''), delimiter=' ')
		consratelist = list(considerusers)
		consratefile.writerows(consratelist)