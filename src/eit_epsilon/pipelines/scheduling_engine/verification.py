from collections import Counter

def verifyJobsMatch(j1, j2):
    print(f"Number of jobs J1: {len(j1)} J2: {len(j2)}" +  " Equal" if len(j1) == len(j2) else " Not equal")

def verifyTasksMatch(j1, j2 , part_to_tasks):
    # Create a dict: key = Length of task seq, value = number of seqs of this length
    # e.g. tasks1[3] gives the number of task sequences of length 3
    tasks1 = {}
    for j in j1:
        l = len(j)
        tasks1[l] = tasks1.get(l, 0) + 1
        
    tasks2= {}
    for jobID, (partID, dueTime) in j2.items():
        l = len(part_to_tasks[partID])
        tasks2[l] = tasks2.get(l, 0) + 1
        
    print("Task lengths " + "match" if tasks1 == tasks2 else "do not match")
    
def verifyMachinesMatch(machList1, machList2):
    print("Machine lists " + "match" if machList1 == machList2 else "do not match")
    
def verifyDurationsMatch(durList, durDict):
    c1 = Counter([str(d) for d in durList])
    
    l = []
    uniquepartIDs = set([partID for (partID, _) in durDict.keys()])
    for partID1 in uniquepartIDs:
        # Get all the durations for the present part
        l1 = [dur for (partID2, task), dur in durDict.items() if partID1 == partID2]
        l.append(str(l1)) # Use the string represention of the list as the key in the Counter dictionary
    c2 = Counter(l)        
    
    print("Durations " + "match" if c1 == c2 else "do not match")
    
    

        