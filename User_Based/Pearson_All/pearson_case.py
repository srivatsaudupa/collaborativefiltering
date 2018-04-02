from pearson import Pearson

def main():
	pearObj = Pearson("test5.txt", 200)
	pearObj.compute(False, True)
	print("test5.txt processed. result5.txt produced")

	pearObj = Pearson("test10.txt", 200)
	pearObj.compute(False, True)
	print("test10.txt processed. result10.txt produced")

	pearObj = Pearson("test20.txt", 200)
	pearObj.compute(False, True)
	print("test20.txt processed. result20.txt produced")

main()