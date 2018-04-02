import csv
import numpy as np
import math
import sys
from iuf import iuf
import time

class Pearson:
	
	def __init__(self, testfile, K):
		self.K = K
		self.TESTFILE = testfile
		if self.TESTFILE == "test5.txt":
			self.UIDOFF = 201
			self.RESULTFILE = "result5.txt"	
		elif self.TESTFILE == "test10.txt":
			self.UIDOFF = 301
			self.RESULTFILE = "result10.txt"	
		else:
			self.UIDOFF = 401
			self.RESULTFILE = "result20.txt"			
		self.TRAINFILE = "train.txt"
		self.trainuavg = {}
	
	def compute(self, isIuf=False, isCaseAmp=False):
		start = time.time()
		trainfile = csv.reader(open(self.TRAINFILE, "r"), delimiter='\t')
		train_data_list = list(trainfile)
		traindata = np.array(train_data_list).astype(int)

		testfile = csv.reader(open(self.TESTFILE, 'r'), delimiter=' ')
		test_data_list = list(testfile)
		testdata = np.array(test_data_list).astype(int)

		if isIuf:
			iuftraindata, iufval = iuf(traindata)
		else:
			iuftraindata = traindata
		# Pearson 
		waumatrix, tstavg = self.compute_pearson(testdata, iuftraindata)

		i= 0

		# k most similar, choose k = 5 
		ksimmatrix = []
		for wau_vect in waumatrix:
			j = 0
			for wau in wau_vect:
				ksimmatrix.append([i+self.UIDOFF, j+1, wau])
				j+=1
			i+=1

		topkmat = self.get_k_nearest(self.K, ksimmatrix)

		testratefile = csv.writer(open("Pearson.csv", "w", newline=''))
		testratelist = list(waumatrix)
		testratefile.writerows(testratelist)

		torate = []

		for test in testdata:
			if(test[2] == 0):
				torate.append(test)
		
		testrate = []

		printAvg = []
		for key, val in self.trainuavg.items():
			for v in val:
				printAvg.append([key, v])
		
		for rate in torate:
			userid = rate[0]
			movid = rate[1]-1
			wausum = 0
			wauprod = 0
			uid = 0
			k = 1
			wau_vect = topkmat[self.UIDOFF]
			for wauval in wau_vect:
				wau = wauval[1]
				if(wau == 0):
					uid += 1
					continue
				avgval = self.trainuavg[userid]
				for avg in avgval:
					if avg[0] == uid:
						ru = avg[1]
						break
				if isCaseAmp:
					wau = wau*(abs(wau))**1.5

				wauprod += wau * (traindata[uid][movid] - ru)
				wausum += abs(wau)
				uid += 1
				k += 1
				if k > self.K:
					break

			if(wausum != 0):
				rating = tstavg[userid] + (wauprod / wausum)
			else:
				rating = tstavg[userid]
			if(rating > 5):
				rating = 5
			elif(rating < 1):
				rating = 1

			testrate.append([userid, movid+1, int(round(rating))])

		testratefile = csv.writer(open(self.RESULTFILE, "w", newline=''), delimiter=' ')
		testratelist = list(testrate)
		testratefile.writerows(testratelist)
		print("Time to execute "+str(time.time()-start))

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


	def compute_pearson(self, testuser, trainuser):
		# Form unique_userid x 1000 matrix
		test_userid = np.unique(testuser[:, 0])
		test_userid_count = len(test_userid)
		test_user_array = np.zeros(shape=(test_userid_count, 1000))

		for test in testuser:
			test_user_array[test[0]-self.UIDOFF][test[1]-1] = test[2]

		tstavg = self.avgmapper(test_user_array, self.UIDOFF)
		#trtavg = self.avgmapper(trainuser, 0)

		wmatrix = np.zeros(shape=(100, 200))
		# Obtain common terms between testuser and trainuser
		test_uid = 0
		for testu in test_user_array:
			train_uid = 0
			self.trainuavg[test_uid+self.UIDOFF] = []
			for trainu in trainuser:
				# wmatrix[test_uid][train_uid] = self.pearson_correlation(testu, trainu)
				wmatrix[test_uid][train_uid], ru = self.compute_correlation(testu, trainu, tstavg[test_uid+self.UIDOFF])
				self.trainuavg[test_uid+self.UIDOFF].append([train_uid, ru])
				train_uid += 1
			test_uid += 1
		return wmatrix, tstavg


	def compute_correlation(self, testu, trainu, ra):
		num, ru, tsttrm, trtrm, rt = 0, 0, 0, 0, 0
		
		movielist = np.array(list(range(1000)))
		testrated=testu!=0
		trainrated=trainu!=0
		commonrated = movielist[[np.bitwise_and(testrated,trainrated)]]
		
		for mov in commonrated:
			ru += trainu[mov]
			rt += 1 
		if(rt == 0):
			return 0, 0

		ru = ru * 1.0 / rt

		for mvi in commonrated:
			rai, rui = testu[mvi], trainu[mvi]
			num+= (rai-ra)*(rui-ru)
			tsttrm += (rai-ra)**2
			trtrm += (rui-ru)**2

		denom = (tsttrm**0.5) * (trtrm)**0.5
		if(denom != 0):
			return num / denom, ru
		return 0, ru

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