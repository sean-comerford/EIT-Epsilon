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
    
def verifyPopulationsMatch(pop1, pop2, croom_processed_orders):
    
    # Loop over the items in this population
    # Example item in a population (taskID, machine, start, duration, task index, part ID):
    # (28, 1, 1, 0, 360.0, 0, 'RIGHT-PS-5N-CTD-OP1')   
    print(f"Len: {len(pop1[0])}")
    for i in range(20):
        print(f"i: {i}", end=" ")
        print(f"Comparing: {pop1[0][i]} {pop2[0][i]}")
        
        # Original pop stores job index as the first item, new structure stores job ID
        # So we need to obtain the job ID for each element in the original population
        jobIndex = pop1[0][i][0]
        jobID1 = croom_processed_orders['Job ID'].iloc[jobIndex]
        
        # Verify job IDs match
        if jobID1 != pop2[0][i][0]:
            print(f"Items at position {i} do not match: {pop1[0][i]} {pop2[0][i]}")
            return False
        
        # Verify other elements match
        # i.e. taskID, machine, start, duration, task index, part ID
        for j in range(1, len(pop1[0][i])):
            if pop1[0][i][j] != pop2[0][i][j]:
                print(f"Items at position {i} do not match: {pop1[0][i]} {pop2[0][i]}")
                return False

    return True
    
    

        