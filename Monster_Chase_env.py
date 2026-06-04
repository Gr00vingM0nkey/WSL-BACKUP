import gymnasium as gym
import numpy as np

class MC:
    def __init__(self, Map_Size):
        self.Map_Size = Map_Size
        self.map = np.full((Map_Size,Map_Size), "0")
        self.map[1,1]="1"#@
        self.map[0,0]="2"#M
        self.turn = 0
        self.action_space = [0,1,2,3] # 0up, 1left 2down 3right
        self.observation_space = np.zeros((1, Map_Size * Map_Size))
        self.stepcount = 0

    def Location(self, map):
        for z in range(self.Map_Size):
            if "1" in map[z]:
                player = [z, map[z].tolist().index("1")]
            if "2" in map[z]:
                monster = [z, map[z].tolist().index("2")]
        return player, monster

    def Move_Up(self, Character_Location):
        Character = self.map[Character_Location[0], Character_Location[1]]
        self.map[Character_Location[0], Character_Location[1]] = "0"
        self.map[max(Character_Location[0] - 1, 0), Character_Location[1]] = Character

    def Move_Down(self, Character_Location):
        Character = self.map[Character_Location[0], Character_Location[1]]
        self.map[Character_Location[0], Character_Location[1]] = "0"
        self.map[min(Character_Location[0] + 1, self.Map_Size - 1), Character_Location[1]] = Character

    def Move_Left(self, Character_Location):
        Character = self.map[Character_Location[0], Character_Location[1]]
        self.map[Character_Location[0], Character_Location[1]] = "0"
        self.map[Character_Location[0], max(Character_Location[1] - 1, 0)] = Character

    def Move_Right(self, Character_Location):
        Character = self.map[Character_Location[0], Character_Location[1]]
        self.map[Character_Location[0], Character_Location[1]] = "0"
        self.map[Character_Location[0], min(Character_Location[1] + 1, self.Map_Size - 1)] = Character

    def Manhatten_Distance(self, player, monster):
        return np.abs((int(player[1])-int(monster[1])))+abs((int(player[0])-int(monster[0])))       
                
    def check_win_lose(self):
        #The reward is Manhatten distance
        #If the Monster is next to the player after the players turn Loss
        observation = self.map.flatten()
        observation = np.where(observation == "1", 1.0, np.where(observation == "2", 2.0, 0.0)).astype(np.float32)
        player, monster = self.Location(self.map)
        distance = (self.Manhatten_Distance(player, monster))**2
        reward = distance if distance != 1 else -10
        terminated = 1 if self.Manhatten_Distance(player, monster) == 1 else 0
        truncated = 1 if self.turn > 50 else 0
        info = None
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.stepcount = 0
        self.map = np.full((self.Map_Size,self.Map_Size), "0")
        self.map[1,1]="1"#@
        self.map[0,0]="2"#M
        self.turn = 0
        obs, _, _, _, _ = self.check_win_lose()
        return obs
    
    def BigMMover(self):
        pass
        """_, monster = self.Location(self.map)
        monster[0]-
        monster[0]-=1
        monster[1]"""
        
    def step(self, action):
        self.stepcount += 1
        self.Player, self.Monster = self.Location(self.map)
        if action == 0:
            self.Move_Up(self.Player)
        if action == 1:
            self.Move_Left(self.Player)
        if action == 2:
            self.Move_Down(self.Player)
        if action == 3:
            self.Move_Right(self.Player)
        state = self.check_win_lose()
        if(state[2] == 0):
            maction = np.random.randint(4, size=1)[0]
            if maction == 0:
                self.Move_Up(self.Monster)
            if maction == 1:
                self.Move_Left(self.Monster)
            if maction == 2:
                self.Move_Down(self.Monster)
            if maction == 3:
                self.Move_Right(self.Monster)

        return self.check_win_lose()
        

#exec(open("Monster_Chase.py").read())