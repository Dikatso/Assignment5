# Dikatso Moshweunyane
# 14 May 2022
# CSC3002 ASSIGN5 Scenario1

import numpy as np
import random
from matplotlib import pyplot, colors
from FourRooms import FourRooms




def main():
    
    # q_table = np.array(
    #         [
    #             # 0   1   2   3   4   5   6   7   8   9  10  11  12
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         ], dtype=np.float32
    #     )
    
    # r_table = np.array(
    #         [
    #             # 0   1   2   3   4   5   6   7   8   9  10  11  12
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         ], dtype=np.float32
    #     )
    
    
    EPOCH = 100
    LEARNING_RATE = 0.7
    DISCOUNT_FACTOR = 0.9
    EPLISON =1
    MAX_EPSILON =1.0
    MIN_EPSILON = 0.01
    DECAY_RATE = 0.01
    
    rewards_track = 0
    # Create FourRooms Object
    fourRoomsObj = FourRooms('simple')
    q_table = np.zeros((13, 13))

    print(q_table)
    
    for epoch in range(EPOCH):
        fourRoomsObj.newEpoch()
        currentPosition = fourRoomsObj.getPosition()
        notDone = True
        rewards_track = 0
        
        while notDone:
            if random.uniform(0, 1) > EPLISON:
                # return np.argmax(q_values[current_row_index, current_column_index])
                nextStep = np.argmax(q_table[currentPosition[0],currentPosition[1]]) 
            else:
                nextStep = random.randint(0, 3)
        
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(nextStep)
            
            # Update Q-table for Q(s,a)
            if q_table[newPos[0], newPos[1]] == 0:
                reward = -1
                
            else:
                reward = q_table[newPos[0], newPos[1]]
            
            q_table[newPos[0], newPos[1]] = q_table[newPos[0], newPos[1]] * (1 - LEARNING_RATE) + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[newPos[0], newPos[1]]))
            if isTerminal:
                notDone = False
                #q_table[newPos[0], newPos[1]] = 100
                print("yeee")
                break
            
        
        normalize_q=q_table/max(q_table[q_table.nonzero()])*100
        normalize_q.astype(int)
        
        print(q_table)
        # print("normalized")
        # print(normalize_q)
        
    fourRoomsObj.showPath(-1)    
        
    # Create FourRooms Object
    # fourRoomsObj = FourRooms('simple')
    
    
    
    # numpy.array(
    #         [
    #             # 0   1   2   3   4   5   6   7   8   9  10  11  12
    #             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 0
    #             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],  # 1
    #             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],  # 2
    #             [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # 3
    #             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],  # 4
    #             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],  # 5
    #             [-1, -1,  0, -1, -1, -1, -1,  0,  0,  0,  0,  0, -1],  # 6
    #             [-1,  0,  0,  0,  0,  0, -1, -1, -1,  0, -1, -1, -1],  # 7
    #             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],  # 8
    #             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],  # 9
    #             [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # 10
    #             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],  # 11
    #             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   # 12
    #         ])
    # This will try to draw a zero
    # actSeq = [FourRooms.LEFT, FourRooms.LEFT, FourRooms.LEFT,
    #           FourRooms.UP, FourRooms.UP, FourRooms.UP,
    #           FourRooms.RIGHT, FourRooms.RIGHT, FourRooms.RIGHT,
    #           FourRooms.DOWN, FourRooms.DOWN, FourRooms.DOWN,FourRooms.LEFT, FourRooms.LEFT, FourRooms.LEFT,
    #           FourRooms.UP, FourRooms.UP, FourRooms.UP,
    #           FourRooms.RIGHT, FourRooms.RIGHT, FourRooms.RIGHT,
    #           FourRooms.DOWN, FourRooms.DOWN, FourRooms.DOWN]

    # aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    # gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']

    # print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))
    # # print(fourRoomsObj.__current_num_packages)
    # for act in actSeq:
    #     gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(act)

    #     print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[act], newPos, gTypes[gridType]))

    #     if isTerminal:
    #         print("yeee")
    #         break

    # # Don't forget to call newEpoch when you start a new simulation run

    # # Show Path
    # fourRoomsObj.showPath(-1)
    # foundIT = tuple()
    
    # while True:
    #     gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(random.randint(0, 3))
    #     print(newPos)
    #     if isTerminal:
    #         foundIT = newPos
    #         break
    
    # print("This is where it is",foundIT)
    # fourRoomsObj.showPath(-1)
if __name__ == "__main__":
    main()
