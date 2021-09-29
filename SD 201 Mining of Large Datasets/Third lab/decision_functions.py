import csv

reader = csv.reader(open('data.csv', newline=''))

list_passengers=[]
#dictionnary with the format "feature":[died,survived]
for row in reader:
    list_passengers.append(row)
del list_passengers[0]
n=len(list_passengers)

def make_records(liste):
    dico={"man":[0,0],"woman":[0,0],"PClass1":[0,0],"PClass2":[0,0],"PClass3":[0,0],"Embarked0":[0,0],"Embarked1":[0,0],"Embarked2":[0,0],"Passengers":liste}
    for passenger in liste:
        if passenger[0]=='0' and passenger[3]=='0':
            dico["woman"][0]+=1
        if passenger[0]=='0' and passenger[3]=='1':
            dico["woman"][1]+=1
        if passenger[0]=='1' and passenger[3]=='0':
            dico["man"][0]+=1
        if passenger[0]=='1' and passenger[3]=='1':
            dico["man"][1]+=1
        if passenger[1]=='1':
            if passenger[3]=='0':
                dico["PClass1"][0]+=1
            else:
                dico["PClass1"][1]+=1
        if passenger[1]=='2':
            if passenger[3]=='0':
                dico["PClass2"][0]+=1
            else:
                dico["PClass2"][1]+=1
        if passenger[1]=='3':
            if passenger[3]=='0':
                dico["PClass3"][0]+=1
            else:
                dico["PClass3"][1]+=1
        if passenger[2]=='0':
            if passenger[3]=='0':
                dico["Embarked0"][0]+=1
            else:
                dico["Embarked0"][1]+=1
        if passenger[2]=='1':
            if passenger[3]=='0':
                dico["Embarked1"][0]+=1
            else:
                dico["Embarked1"][1]+=1
        if passenger[2]=='2':
            if passenger[3]=='0':
                dico["Embarked2"][0]+=1
            else:
                dico["Embarked2"][1]+=1
    return dico

                    
def Node(nature,level,feature,children,gini,classe_value,record):
    #nature=string
    #feature=["attribute",constraint]
    return [nature,level,feature,children,gini,classe_value,record]
 

records=make_records(list_passengers)
list_nodes=[]
tot=records["man"][0]+records["woman"][0]+records["man"][1]+records["woman"][1]
p0_init=(records["man"][0]+records["woman"][0])/tot
p1_init=(records["man"][1]+records["woman"][1])/tot
gini_init=1-(p0_init**2)-(p1_init**2)

def BuildDecisionTree(minNum,reader = csv.reader(open('data.csv', newline=''))):
    #D is a list of records
    global list_nodes
    list_nodes=[]
    def Build(D,minNum,node):
        global errors
        global list_nodes
        test=0
        d=D["Passengers"]
        for k in range(1,len(d)):
            if int(d[k-1][3])!=int(d[k][3]):
                test=2
            if len(d)<minNum:
                test=1
        if test==0:
            #All the records have the same class)
            node=Node("Leaf",node[1],node[2],[],node[4],d[0][3],d)          
            return list_nodes.append(node)
        elif test==1:
            node=Node("Leaf",node[1],node[2],[],node[4],"0",d) #Node becomes leaf with default class '0'
            return list_nodes.append(node)
        else:
            temp=gini_split(D)
            temp2=D["Passengers"]
            if temp[2]==[1,1,1,1,1]:
                #then it is a leaf
                c0=0
                c1=0
                for passenger in temp2:
                    if passenger[3]=='0':
                        c0+=1
                    else:
                        c1+=1
                if c0>c1:
                    node=Node("Leaf",node[1],node[2],[],node[4],"0",d)
                else:
                    node=Node("Leaf",node[1],node[2],[],node[4],"1",d)
                list_nodes.append(node)
            else:
                d1=[]
                d2=[]
                if temp[0]=='Sex':
                    for passenger in temp2:
                        if passenger[0]=='0':
                            d1.append(passenger)
                        else:
                            d2.append(passenger)
                            alt='1'
                if temp[0]=='PClass':
                    if temp[1]==2:
                        for passenger in temp2:
                            if passenger[1]=='1':
                                d1.append(passenger)
                            else:
                                d2.append(passenger)
                                alt='>=2'
                    else:
                        for passenger in temp2:
                            if int(passenger[1])<3:
                                d1.append(passenger)
                            else:
                                d2.append(passenger)
                                alt='3'
                if temp[0]=='Embarked':
                    if temp[1]==1:
                        for passenger in temp2:
                            if passenger[2]=='0':
                                d1.append(passenger)
                            else:
                                d2.append(passenger)
                                alt='>=1'
                    else:
                        for passenger in temp2:
                            if int(passenger[2])<2:
                                d1.append(passenger)
                            else:
                                d2.append(passenger)
                                alt='2'
                D1=make_records(d1)
                D2=make_records(d2)
                node1=Node("Intermediate",node[1]+1,temp[:2],[],gini(D,[None,None,temp])[0],None,D1["Passengers"])
                node2=Node("Intermediate",node[1]+1,[temp[0],alt],[],gini(D,[None,None,temp])[1],None,D2["Passengers"])
                #node[3].append(node1)
                #node[3].append(node2)
                list_nodes.append(node)
                list_nodes[len(list_nodes)-1][2]=temp[:2]
                Build(D1,minNum,node1)
                Build(D2,minNum,node2)
               

    return Build(records,minNum,["Root",0,None,[],gini_init,None,records["Passengers"]]),list_nodes

        
def get_indexes_min_value(l):
        min_value = min(l)
        if l.count(min_value) > 1:
             return [i for i, x in enumerate(l) if x == min(l)]
        else:
             return l.index(min(l))

       
def gini(dico,node):
    dusbin=0
    if node[2][0]=="Sex":
        men=dico["man"][0]+dico["man"][1]
        women=dico["woman"][0]+dico["woman"][1]
        if(men==0 or women==0):
            dusbin=1
        else:
            gini0=1-((dico["man"][0]/men)**2)-((dico["man"][1]/men)**2)
            gini1=1-((dico["woman"][0]/women)**2)-((dico["woman"][1]/women)**2)
            return [gini1,gini0,women,men]       
    if node[2][0]=="PClass":
        if node[2][1]==2:
            size=dico["PClass1"][0]+dico["PClass1"][1]
            size2=dico["PClass2"][0]+dico["PClass2"][1]+dico["PClass3"][0]+dico["PClass3"][1]
            if (size==0 or size2==0):
                dusbin=1
            else:
                gini0=1-((dico["PClass1"][0]/size)**2)-((dico["PClass1"][1]/size)**2)
                p0=dico["PClass2"][0]+dico["PClass3"][0]
                p1=dico["PClass2"][1]+dico["PClass3"][1]
                gini1=1-((p0/size2)**2)-((p1/size2)**2)
                return [gini0,gini1,size,size2]
        if node[2][1]==3:
            size=dico["PClass1"][0]+dico["PClass1"][1]+dico["PClass2"][0]+dico["PClass2"][1]
            size2=dico["PClass3"][0]+dico["PClass3"][1]
            if(size==0 or size2==0):
                dusbin=1
            else:
                p0=dico["PClass1"][0]+dico["PClass2"][0]
                p1=dico["PClass1"][1]+dico["PClass2"][1]
                gini0=1-((p0/size)**2)-((p1/size)**2)
                gini1=1-((dico["PClass3"][0]/size2)**2)-((dico["PClass3"][1]/size2)**2)
                return [gini0,gini1,size,size2]
    if node[2][0]=="Embarked":
        if node[2][1]==1:
            size=dico["Embarked0"][0]+dico["Embarked0"][1]
            size2=dico["Embarked1"][0]+dico["Embarked1"][1]+dico["Embarked2"][0]+dico["Embarked2"][1]
            if (size==0 or size2==0):
                dusbin=1
            else:
                gini0=1-((dico["Embarked0"][0]/size)**2)-((dico["Embarked0"][1]/size)**2)
                p0=dico["Embarked1"][0]+dico["Embarked2"][0]
                p1=dico["Embarked1"][1]+dico["Embarked2"][1]
                gini1=1-((p0/size2)**2)-((p1/size2)**2)
                return [gini0,gini1,size,size2]
        if node[2][1]==2:
            size=dico["Embarked0"][0]+dico["Embarked0"][1]+dico["Embarked1"][0]+dico["Embarked1"][1]
            size2=dico["Embarked2"][0]+dico["Embarked2"][1]
            if(size==0 or size2==0):
                dusbin=1
            else:
                p0=dico["Embarked0"][0]+dico["Embarked1"][0]
                p1=dico["Embarked0"][1]+dico["Embarked1"][1]
                gini0=1-((p0/size)**2)-((p1/size)**2)
                gini1=1-((dico["Embarked2"][0]/size2)**2)-((dico["Embarked2"][1]/size2)**2)
                return [gini0,gini1,size,size2]
        
def gini_split(dico):
    giniSplit=[]
    a=gini(dico,[None,None,["Sex",0]])
    if type(a)==type(None):
        giniSplit.append(1)
    else:
        giniSplit.append((a[0]*a[2]/(a[2]+a[3]))+(a[1]*a[3]/(a[2]+a[3])))
    a=gini(dico,[None,None,["PClass",2]])
    if type(a)==type(None):
        giniSplit.append(1)
    else:
        giniSplit.append((a[0]*a[2]/(a[2]+a[3]))+(a[1]*a[3]/(a[2]+a[3])))
    a=gini(dico,[None,None,["PClass",3]])
    if type(a)==type(None):
        giniSplit.append(1)
    else:
        giniSplit.append((a[0]*a[2]/(a[2]+a[3]))+(a[1]*a[3]/(a[2]+a[3])))
    a=gini(dico,[None,None,["Embarked",1]])
    if type(a)==type(None):
        giniSplit.append(1)
    else:
        giniSplit.append((a[0]*a[2]/(a[2]+a[3]))+(a[1]*a[3]/(a[2]+a[3])))
    a=gini(dico,[None,None,["Embarked",2]])
    if type(a)==type(None):
        giniSplit.append(1)
    else:
        giniSplit.append((a[0]*a[2]/(a[2]+a[3]))+(a[1]*a[3]/(a[2]+a[3])))
    i=get_indexes_min_value(giniSplit)
    if type(i)==type([]):
        i=i[0]
    if i==0:
        return ["Sex",0,giniSplit]
    if i==1:
        return ["PClass",2,giniSplit]
    if i==2:
        return ["PClass",3,giniSplit]
    if i==3:
        return ["Embarked",1,giniSplit]
    if i==4:
        return ["Embarked",2,giniSplit]

def printDecisionTree(tree,file):
    fichier = open(file, "w")
    list_nodes=tree
    levels=[node[1] for node in list_nodes]
    m=max(levels)
    l=[[]]
    for k in range(m):
        l+=[[]]
    for node in list_nodes:
        l[node[1]].append(node)
    for i in range(m+1):
        for node in l[i]:
            print(node[0])
            fichier.write("\n"+node[0])
            print("Level"+" "+ str(node[1]))
            fichier.write("\n"+"Level"+" "+ str(node[1]))
            if node[0]=='Leaf':
                print("Class"+" "+node[5])
                fichier.write("\n"+"Class"+" "+node[5])
            else:
                if node[2][0]=="PClass":
                    if node[2][1]==3:
                        print("Feature"+" "+node[2][0]+" "+"1 2")
                        fichier.write("\n"+"Feature"+" "+node[2][0]+" "+"1 2")
                    if node[2][1]==2:
                        print("Feature"+" "+node[2][0]+" "+"1")
                        fichier.write("\n"+"Feature"+" "+node[2][0]+" "+"1")
                elif node[2][0]=="Embarked":
                    if node[2][1]==1:
                        print("Feature"+" "+node[2][0]+" "+"0")
                        fichier.write("\n"+"Feature"+" "+node[2][0]+" "+"0")
                    elif node[2][1]==2:
                        print("Feature"+" "+node[2][0]+" "+"0 1")
                        fichier.write("\n"+"Feature"+" "+node[2][0]+" "+"0 1")
                else:
                    print("Feature"+" "+node[2][0]+" "+str(node[2][1]))
                    fichier.write("\n"+"Feature"+" "+node[2][0]+" "+str(node[2][1]))
            print("Gini"+" "+str(node[4]))
            fichier.write("\n"+"Gini"+" "+str(node[4]))
            if l[i].index(node)<len(l[i])-1:
                print("*****")
                fichier.write("\n"+"*****")
        print("")
        fichier.write("\n ")
    fichier.close()
    return

def generalizationError(alpha,tree):
    nb_leaves=0
    list_nodes2=tree
    e=0
    for node in list_nodes2:
        if node[0]=="Leaf":
            nb_leaves+=1
            for passenger in node[6]:
                if passenger[3]!=node[5]:
                    e+=1
    return (e+alpha*nb_leaves)

def pruneTree(minNum,alpha,tree):
    temp=tree
    levels=[node[1] for node in tree]
    m=max(levels)
    l=[[]]
    for k in range(m):
        l+=[[]]
    for node in tree:
        l[node[1]].append(node)
    for i in range(m+1):
        for node in l[m-i]:
            temp2=[]
            for nodes in temp:
                temp2.append(nodes)
            if node[0]!="Leaf":
                i=temp.index(node)
                #we remove the children
                k=1
                while(i+k<len(temp) and temp[i+k][1]>node[1]):
                    del temp2[i+1]
                    k+=1
                #we transform the node in leaf
                d=node[6]
                c0=0
                c1=0
                for passenger in d:
                    if passenger[3]=='0':
                        c0+=1
                    else:
                        c1+=1
                if c0==len(d) or c1==len(d):
                    #all the records have the same class
                    temp2[i]=Node("Leaf",node[1],node[2],[],node[4],d[0][3],d)
                if len(d)<minNum or c0==c1:
                    temp2[i]=Node("Leaf",node[1],node[2],[],node[4],"0",d)
                if c1>c0:
                    temp2[i]=Node("Leaf",node[1],node[2],[],node[4],"1",d)
                if c0>c1:
                    temp2[i]=Node("Leaf",node[1],node[2],[],node[4],"0",d)
                if generalizationError(alpha,temp2)<generalizationError(alpha,temp):
                    temp=temp2
    printDecisionTree(temp,"postprunned_tree.txt")
    