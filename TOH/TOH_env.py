import gymnasium as gym
import numpy as np

class Tower_of_Hanoi(gym.Env):

    def __init__(self, height:int = 3, repeatcon:int=0):
        self.Repeat_Con = repeatcon  
        self.Reward_Addition = 0
        self.Repeat_Num = np.array([])
        self.HEIGHT = height
        self.Check = False
        self.Past_disks = np.array([])
        self.Least_Moves = np.array((2**self.HEIGHT)-1)
        self.Tower = np.array(np.array((np.flip(np.arange(1, self.HEIGHT+1))))),np.array([]),np.array([])
        print(self.Tower)
        self.Check_Tower = np.array([np.flip(np.arange(self.HEIGHT))])
        self.Moves = np.array([])
        self.observation_space = gym.spaces.Box(0, 1, shape=(3,1))
        #The actions.
        # 1->2=0, 1->3=1, 2->1=2, 2->3=3, 3->1=4, 3->2=5
        self.action_space = gym.spaces.Discrete(5)
    
    def Check_Win(self, T3):
        return T3==self.check_tower
    
    def Reward(self):
        #EQUATION = (2^n)-1 of total moves
        #EXAMPLE HEIGHT IS 4 SO 15 moves minimum
        #____REWARD IDEA
        #give it a reward based on the size of the disk on the third pole
        #big disk big reward, small disk small reward
        #dont give it reward for repeating moves. 
        #Try out both half then quarter then no reward with repeated moves. 
        T3 = self.Tower[2]
        T3_Disks = list(str(T3))
        for z in T3_Disks:
            pot_reward = z
            if z not in self.Past_disks:
                self.Past_disks.append(z)
                return 0
            if z in self.Past_disks:
                pot_reward *= (self.Repeat_Con)**(self.Past_disks.count(z))#this applies the repeated discount
                return int(pot_reward)
                
    def Output(self,state):
        #0-Continue, 1-Win, 2-Lose, 3-ILLEGAL MOVE
        term = False
        reward = 0
        match state:
            case 0:
                reward += self.Reward_Addition + self.Reward(self.Tower[2])# -_+
            case 1:
                reward += self.Reward_Addition + self.Reward(self.Tower[2])#-_+
                term = True
            case 2:
                term = True
            case 3:
                reward += -1
        #obs, reward, term, trun, info
        obs = self.Tower
        for z in range(self.HEIGHT):
            if(len(obs[z])!=self.HEIGHT):
                for i in range(self.HEIGHT-len(obs[z])):
                    np.insert(obs[z],0)
        for z in range(3):
            obs[z] = np.array(obs[z])
        obs = np.array(obs)
        return obs, reward, term, False, False

    def Check_Repetition(self):
        move = self.Moves[-1]
        for z in range(4):
            if(move==self.Moves[-1]):
                self.Reward_Addition += -1


    def reset(self, height:int=3, repeatcon:int=0):
        self.Repeat_Con = repeatcon  
        self.Reward_Addition = 0
        self.Repeat_Num = np.array([])
        self.HEIGHT = height
        self.Check = False
        self.Past_disks = np.array([])
        self.Least_Moves = np.array((2**self.HEIGHT)-1)
        self.Tower = np.array(np.array((np.flip(np.arange(1, self.HEIGHT+1))))),np.array([]),np.array([])
        print(self.Tower)
        self.Check_Tower = np.array([np.flip(np.arange(self.HEIGHT))])
        self.Moves = np.array([])
        return self.Output(2)

    def Check_Legal(self, action):
        # 1->2=0, 1->3=1, 2->1=2, 2->3=3, 3->1=4, 3->2=5
        match action:
            case 0:#T1 MOVES
                return self.Tower[0][-1]>self.Tower[1][-1]
            case 1:
                return self.Tower[0][-1]>self.Tower[2][-1]
            
            case 2:#T2 MOVES 
                return self.Tower[1][-1]>self.Tower[0][-1]
            case 3:
                return self.Tower[1][-1]>self.Tower[2][-1]
            
            case 4:#T3 MOVES
                return self.Tower[2][-1]>self.Tower[0][-1]
            case 5:
                return self.Tower[2][-1]>self.Tower[1][-1]

    def step(self, action):
        # 1->2=0, 1->3=1, 2->1=2, 2->3=3, 3->1=4, 3->2=5
        if(self.Check_Legal(action)):#LEGAL ACTION
            self.Moves.append(action)
            match action:
                case 0:#T1 MOVES
                    self.Tower[1] = np.append(self.Tower[1], self.Tower[0][-1])
                    #self.Tower[1].insert(self.Tower[0][-1])
                    self.Tower[0][-1].delete
                case 1:
                    self.Tower[2].pop(self.Tower[0][-1])
                
                case 2:#T2 MOVES
                    self.Tower[0].pop(self.Tower[1][-1])
                case 3:
                    self.Tower[2].pop(self.Tower[1][-1])
                
                case 4:#T3 MOVES
                    self.Tower[0].pop(self.Tower[2][-1]) 
                case 5:
                    self.Tower[1].pop(self.Tower[2][-1])
            if(len(self.Moves)>(5*int(self.Least_Moves))):
                return self.Output(2)
            if(self.Check_Win(self.Tower[2])):
                return self.Output(1)
            return self.Output(0)
        else:#ILLEGAL ACTION
            return self.Output(3)
               
env=Tower_of_Hanoi()
_,_,_,_,_ = env.reset()