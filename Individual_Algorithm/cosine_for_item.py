import csv
import numpy as np
import math
import sys
import time

class Cosine_For_Item:

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

	def compute(self, caseamp=False):
		start = time.time()
		trainfile = csv.reader(open(self.TRAINFILE, "r"), delimiter='\t')
		train_data_list = list(trainfile)
		traindata = np.array(train_data_list).astype(int)

		trtavg = self.avgmapper(traindata, 0)

		adjtraindata = np.zeros(shape=(200, 1000))
		
		uid = 0
		for train in traindata:
			movid = 0
			for rate in train:
				if rate != 0:
					newrate = rate-trtavg[uid]
					adjtraindata[uid][movid] = newrate
				movid += 1
			uid += 1


		#traindata = traindata.T
		
		testratefile = csv.writer(open(str(self.UIDOFF)+".csv", "w", newline=''))
		testratelist = list(adjtraindata)
		testratefile.writerows(testratelist)

		adjtraindata = adjtraindata.T


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

		test_user_array = test_user_array.T

		movie_to_rate = {}

		for test in testdata:
			movie_to_rate[test[1]] = []

		movie_to_rate = self.sortByCosim(self.compute_similarity(movie_to_rate, adjtraindata))

		# Each test user rated movies
		toRate = []
		for test in testdata:
			if(test[2] == 0):
				toRate.append(test)

		testrate = []
		for rate in toRate:
			userid = rate[0]
			movid = rate[1]
			cosimmat = movie_to_rate[movid]
			rating = 0
			wsum = 0
			k = 0
			for cos in cosimmat:
				matmovid = cos[0]
				if(test_user_array[matmovid-1][userid-self.UIDOFF] != 0):
					rating += abs(cos[1]) * test_user_array[matmovid-1][userid-self.UIDOFF]
					wsum += abs(cos[1])
					k += 1
					if k > self.K:
						break

			if wsum != 0:
				userrating = int(round(rating/wsum))
			else:
				userrating = int(round(testsum[userid-self.UIDOFF]*1.0 / self.MOVIECOUNT))

			if userrating > 5:
				userrating =5
			elif userrating < 1:
				userrating = 1

			testrate.append([userid, movid, userrating])

		testratefile = csv.writer(open("item_"+self.RESULTFILE, "w", newline=''), delimiter=' ')
		testratelist = list(testrate)
		testratefile.writerows(testratelist)


	def compute_similarity(self, movie_to_rate, traindata):
		
		for mov, simvect in movie_to_rate.items():
			for i in range(1000):
				if i+1 != mov:
					cosim = self.compute_cosine_similarity(traindata[mov-1], traindata[i])
					movie_to_rate[mov].append([i+1, cosim])
		return movie_to_rate

	def compute_cosine_similarity(self, testvector, trainvector, caseamp = False):
		
		testhasrated = testvector!=0
		trainhasrated = trainvector!=0
		movielist = np.array(list(range(200)))
		commonrated = movielist[[np.bitwise_and(testhasrated,trainhasrated)]]
		if len(commonrated) == 0:
			return 0

		dotprod, trainsqr, testsqr = 0,0,0
		for mov in commonrated:
			dotprod += testvector[mov]*trainvector[mov]
			trainsqr += trainvector[mov]**2
			testsqr += testvector[mov]**2
		return dotprod / (np.sqrt(trainsqr)*np.sqrt(testsqr))


	def sortByCosim(self, topkmat):

		for key, vals in topkmat.items():
			newvals = []
			vals.sort(key = lambda x:x[1], reverse=True)
			del topkmat[key]
			topkmat[key] = vals
		return topkmat

	def avgmapper(self, testuser, initid):
		tstavg = {}
		testuid = initid
		for testu in testuser:
			tst, tsta = 0, 0
			for testrat in testu:
				if(testrat == 0):
					continue
				tsta += testrat
				tst += 1
			tsta = tsta * 1.0 / tst
			tstavg[testuid] = tsta 
			testuid += 1
		return tstavg