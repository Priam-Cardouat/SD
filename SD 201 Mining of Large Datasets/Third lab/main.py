try:
	from decision_functions import BuildDecisionTree
	BuildDecisionTree(minNum=5)
	print('BuildDecisionTree loaded!')
	print('----')
except Exception as e:
	raise e

try:
	from decision_functions import printDecisionTree
	printDecisionTree(BuildDecisionTree(minNum=5)[1],"output_tree.txt")
	print('printDecisionTree loaded!')
	print('----')
except Exception as e:
	raise e

try:
    from decision_functions import generalizationError
    print(generalizationError(alpha=0.5,tree=BuildDecisionTree(minNum=5)[1]))
    fichier = open("generalization_error.txt", "w")
    fichier.write(str(generalizationError(alpha=0.5,tree=BuildDecisionTree(minNum=5)[1])))
    fichier.close()
    print('generalizationError loaded!')
    print('----')
except Exception as e:
	raise e

try:
	from decision_functions import pruneTree
	pruneTree(minNum=5,alpha=0.5,tree=BuildDecisionTree(minNum=5)[1])
	print('pruneTree loaded!')
	print('----')
except Exception as e:
	raise e