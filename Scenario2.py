# Dikatso Moshweunyane
# 14 May 2022
# CSC3002 ASSIGN5 Scenario1

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot, colors
from FourRooms import FourRooms


def main():
    
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
    EPOCH = 50
    LEARNING_RATE = 0.9
    DISCOUNT_FACTOR = 0.9
    EPLISON = 1
    MAX_EPSILON =1.0
    MIN_EPSILON = 0.01
    DECAY_RATE = 0.01
    
    rewards_track = 0
    # Create FourRooms Object
    fourRoomsObj = FourRooms('multi')
    global_start = fourRoomsObj.getPosition()
    
    for epoch in range(EPOCH):
        fourRoomsObj.newEpoch()
        currentPosition = fourRoomsObj.getPosition()
        notDone = True
        rewards_track = 0
        while notDone:
            if random.uniform(0, 1) > EPLISON:
                nextStep = np.argmax(q_table[currentPosition,:]) 
            else:
                nextStep = random.randint(0, 3)
        
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(nextStep)
            
            currentPosition = newPos
            
            reward = r_table[currentPosition[1],currentPosition[0]]
            q_table[newPos[1], newPos[0]] = q_table[newPos[1], newPos[0]] * (1 - LEARNING_RATE) + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[newPos[1], newPos[0]]))
            
            r_table[currentPosition[1],currentPosition[0]] = -1
            
            if isTerminal:
                notDone = False
                r_table[currentPosition[1],currentPosition[0]] = 100
                break
        EPLISON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * epoch)   
    
        
    fourRoomsObj.showPath(-1) 
if __name__ == "__main__":
    main()
