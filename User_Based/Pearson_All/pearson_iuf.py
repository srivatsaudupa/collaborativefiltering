from pearson import Pearson
import sys
def main():

	if len(sys.argv) == 1:
		K = 200
	else:
		K = int(sys.argv[1])

	pearObj = Pearson("test5.txt", K)
	pearObj.compute(True, False)
	print("test5.txt processed. result5.txt produced")

	pearObj = Pearson("test10.txt", K)
	pearObj.compute(True, False)
	print("test10.txt processed. result10.txt produced")

	pearObj = Pearson("test20.txt", K)
	pearObj.compute(True, False)
	print("test20.txt processed. result20.txt produced")

main()