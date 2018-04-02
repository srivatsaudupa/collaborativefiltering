import csv
import numpy as np 

class Compute:
	def __init__(self, userfile):
		self.userfile = userfile
		if userfile == "cosine_result5.txt":
			self.itemfile = "item_result5.txt"
			self.resultfile = "result5.txt"
		elif userfile == "cosine_result10.txt":
			self.itemfile = "item_result10.txt"
			self.resultfile = "result10.txt"
		else:
			self.itemfile = "item_result20.txt"
			self.resultfile = "result20.txt"

	def compute(self):
		read_userfile = csv.reader(open(self.userfile, "r"), delimiter=' ')
		user_data_list = list(read_userfile)
		userdata = np.array(user_data_list).astype(int)
		
		read_itemfile = csv.reader(open(self.itemfile, "r"), delimiter=' ')
		item_data_list = list(read_itemfile)
		itemdata = np.array(item_data_list).astype(int)
		
		combinedata = (0.6*userdata[:, 2]+0.4*itemdata[:, 2])
		
		testrate = []

		i = 0
		
		for user in userdata:
			testrate.append([user[0], user[1], int(round(combinedata[i]))])
			i+=1

		testratefile = csv.writer(open(self.resultfile, "w", newline=''), delimiter=' ')
		testratelist = list(testrate)
		testratefile.writerows(testratelist)


def main():
	comp = Compute("cosine_result5.txt")
	comp.compute()
	comp = Compute("cosine_result10.txt")
	comp.compute()
	comp = Compute("cosine_result20.txt")
	comp.compute()

main()

