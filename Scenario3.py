# Dikatso Moshweunyane
# 14 May 2022
# CSC3002 ASSIGN5 Scenario2

import numpy as np
import random
from matplotlib import pyplot, colors
from FourRooms import FourRooms


def main():
    # Initializes the q and reward tables
    q_table = np.array(
            [
                # 0   1   2   3   4   5   6   7   8   9  10  11  12
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100], 
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
            ], dtype=np.float32
        )
    
    r_table = np.array(
            [
                # 0   1   2   3   4   5   6   7   8   9  10  11  12
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=np.float32
        )
    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']
    
    #Parameters to be used in the Q-learning calculations
    EPOCH = 50
    LEARNING_RATE = 0.9
    DISCOUNT_FACTOR = 0.9
    EPLISON = 1
    MAX_EPSILON =1.0
    MIN_EPSILON = 0.01
    DECAY_RATE = 0.01
    
    rewards_track = 0
    
    # Create FourRooms Object and check what the user wants# Create FourRooms Object
    
    checkInput = eval(input("Enter a number: \n1 - Deterministic \n2 - Stochastic \n"))
    
    if checkInput == 1:
        fourRoomsObj = FourRooms('rgb')
    
    else:
        fourRoomsObj = FourRooms('rgb', True)
    
    global_start = fourRoomsObj.getPosition()
    
    for epoch in range(EPOCH):
        fourRoomsObj.newEpoch()
        currentPosition = fourRoomsObj.getPosition()    #Gets the agents intial position
        notDone = True
        rewards_track = 0
        packagesRemaining = 3
        gridType = 0
        
        color_tracker = 0
        gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']
        
        while notDone:  
            if random.uniform(0, 1) > EPLISON:  #Agent checks if it should explore or exploit
                nextStep = np.argmax(q_table[currentPosition,:]) 
            else:
                nextStep = random.randint(0, 3)

            if packagesRemaining == 2 and gTypes[gridType] == "BLUE":
                continue
            
            if packagesRemaining == 3 and gTypes[gridType] == "GREEN":
                continue
            
            if packagesRemaining == 1 and gTypes[gridType] == "RED":
                continue
            
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(nextStep)
            
            currentPosition = newPos
            
            reward = r_table[currentPosition[1],currentPosition[0]]
            #Q-table is being updated
            q_table[newPos[1], newPos[0]] = q_table[newPos[1], newPos[0]] * (1 - LEARNING_RATE) + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[newPos[1], newPos[0]]))
            
            r_table[currentPosition[1],currentPosition[0]] = -1 #-1 rewards for non terminal states
        
            
            
            
            if isTerminal:
                notDone = False
                r_table[currentPosition[1],currentPosition[0]] = 100
                break
        EPLISON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * epoch)       #Calculating new EPSILON value
    
        
    fourRoomsObj.showPath(-1) #Prints out path
if __name__ == "__main__":
    main()
