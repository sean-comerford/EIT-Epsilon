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
        
    print("Task lengths " + ("match" if tasks1 == tasks2 else "do not match"))
    
def verifyMachinesMatch(machList1, machList2):
    print("Machine lists " + ("match" if machList1 == machList2 else "do not match"))
    
def verifyDurationsMatch(durList, jobs2, durDict):
    c1 = Counter([str(d) for d in durList])   

    durList2 = []    
    for jobID, (partID, _) in jobs2.items():
        l = [d for (pId, _), d in durDict.items() if pId == partID]
        durList2.append(l)    
    
    c2 = Counter([str(d) for d in durList2])    
    print("Durations " + ("match" if c1 == c2 else "do not match"))
    
    

        