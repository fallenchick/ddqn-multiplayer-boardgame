"""The Double DQN agent.

Three design points worth reading before the code:

1. **Action masking.**  The legal move set is state-dependent — it depends on
   the turn phase *and* on card holdings.  ``get_mask`` encodes it as a 0/1
   vector over the 101 actions; illegal actions are pushed to -1e5 before the
   argmax, so they are never selected and contribute no gradient signal.

2. **Delayed ticket payoff.**  Ticket value is booked only at the end of the
   episode (``calc_end_bonus_score``) while route completion is rewarded
   immediately.  With gamma = 1 this makes the agent plan for future route
   acquisition instead of grabbing short-term value.

3. **Unreliable loss.**  Because masked actions contribute zero loss, states
   with fewer legal actions have systematically smaller loss; combined with
   stochastic replay sampling, the recorded MSE swings between 0.41 and ~1600
   within one run.  The loss curve is therefore *not* used as a convergence
   signal — see the README for the evaluation protocol that replaces it.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from . import config, env
from .env import Player, Ticket, can_buy, complete_buy_route, pathfinder
from .network import FNN

def _log(*args, **kwargs):
    """Debug output, silenced unless ``config.VERBOSE`` is on.

    The original script printed every turn, which dominated the runtime of a
    long training run and buried the actual results.  Routing all of it
    through this helper makes the default output clean and quiet.
    """
    if config.VERBOSE:
        print(*args, **kwargs)


class DDQNPlayer(Player):
    def __init__(self,name='DDQN'):
        super().__init__(name)
        self.name=name
        self.status=3.5
        self.done=0
        self.batch_size=32
        self.lr=1e-3
        self.e_greed=1
        self.to_buy_route_index=-1#temp
        self.gamma=1#time series depre. reward rate
        #len(state)=142+config.PLAYER_NUM*2
        self.online_net=FNN(config.STATE_DIM_BASE + config.STATE_DIM_PER_PLAYER * config.PLAYER_NUM,
                    len(env.game.route_list) + 1)
        self.target_net=FNN(config.STATE_DIM_BASE + config.STATE_DIM_PER_PLAYER * config.PLAYER_NUM,
                    len(env.game.route_list) + 1)
        self.opt=optim.Adam(self.online_net.parameters(),lr=self.lr)
        self.memory=deque(maxlen=1000)#1000
    def DDQNinit(self,eps):
        super().__init__(self.name)
        if eps<config.EPS_WARMUP:
            return
        self.lr=1e-3/int((eps+1)**(1/3))#changeable lr
        for param_group in self.opt.param_groups:
            param_group['lr']=self.lr                
        self.e_greed=max(self.e_greed*0.998,0.01)#can change to 1-1e6
    def save_current_state(self,tickets=False):#use after game init
        state=[self.status]
        for i in range(len(env.game.route_list)):
            state.append(env.game.route_list[i].controller)#every route(100 routes) and controller
        temp=[0]*env.game.TICKET_NUM
        for ticket in self.tickets:
            temp[ticket.index]=-ticket.value
        for ticket in self.finished_tickets:
            temp[ticket.index]=ticket.value#tickets owned.Gradient to VP=1
        state.append(temp)
        for i in range(len(env.game.players)):
            if env.game.players[i]==self:
                state.append(i)#sequence
        for i in range(len(env.game.players)):
            state.append((len(env.game.players[i].tickets),len(env.game.players[i].cards)))#player info i=0-3
        if tickets:
            state.append(tickets)
        else:
            state.append((len(env.game.ticket_list),-1,-1,-1))#3 tickets -7:-10
        state.append(len(env.game.draw_pile))#env.game.draw_pile cards num -6
        state.append(env.game.get_show_pile())#env.game.show_pile[0,1,2,3,4] -1:-5
        flattened=[]
        for item in state:
            if isinstance(item,list) or isinstance(item,tuple):
                flattened.extend(item)
            else:
                flattened.append(item)
        return flattened
    def select_action(self,state):
        state=torch.tensor(state).to(torch.float32)
        mask=self.get_mask(state)
        legal_actions=np.where(np.arange(len(mask))*np.array(mask)>0)[0]#may not have legal actions
        if len(legal_actions)==0:
            return 100
        temp=np.random.rand()
        if not config.LEARN:
            self.e_greed=-1#evaluate
        if temp<self.e_greed:
            action=random.choice(legal_actions)#random_action
        else:
            with torch.no_grad():
                q_values = self.online_net(state)#输出所有action的Q值
                q_values-=1e5*(1-mask)
                action = int(np.argmax(q_values))
                _log("DDQN status:",self.status,"DDQN Action:",action)
        return action
    def get_mask(self,state):
        state=np.array(state)
        mask=np.zeros(len(env.game.route_list)+1)#100 index100=wait
        if self.status==-1:
            mask[100]=1
        elif self.status==0:
            if state[-6]>3:#sum(env.game.get_show_pile())!=-5 or state[-6]>0:
                mask[1]=1
            for route in env.game.route_list:
                if can_buy(self,route):
                    mask[2]=1
                    break
            if state[-10]>5 and self.name!='dummy':
                mask[3]=1
        elif self.status==1:
            temp=state[-5:]#show_pile[0,0,0,0,0]
            for color in temp:
                if color!=-1:
                    mask[int(color)]=1#color here is float in np
            if state[-6]>3:#len of draw_pile
                mask[9]=1
        elif self.status==1.5:
            temp=state[-5:]
            for color in temp:
                if color!=-1:
                    mask[int(color)]=1
            if state[-6]>0:#len of draw_pile
                mask[9]=1
            mask[0]=0
        elif self.status==2:
            for i in range(len(env.game.route_list)):
                if can_buy(self,env.game.route_list[i]):
                    mask[i]=1
        elif self.status==2.5:
            route=env.game.route_list[self.to_buy_route_index]
            player_have=self.cards_visualizer()
            for i in range(1,9):
                if player_have[0]+player_have[i]>=route.length:
                    mask[i]=1
        elif self.status==3:
            if state[-10]==1:
                mask[1]=1
            elif state[-10]==2:
                mask[1:4]=1
            else:
                mask[1:8]=1
        elif state[0]==3.5:
            mask[4:8]=1
        return mask
    def learn(self):#unfinished
        if len(self.memory)<config.MEMORY_WARMUP:
            return
        state,action,reward,next_state,done=self.sample()#batched-32
        state=torch.tensor(state).to(torch.float32)
        action=torch.tensor(action).to(torch.int64)
        reward=torch.tensor(reward).to(torch.float32)
        next_state=torch.tensor(next_state).to(torch.float32)
        done=torch.tensor(done).to(torch.float32)

        current_q=self.online_net(state).gather(1,action.unsqueeze(1))
        next_action=torch.argmax(self.online_net(next_state),dim=1)#batched
        target_q=self.target_net(next_state).gather(1,next_action.unsqueeze(1))#32*1
        temp=((1-done)*self.gamma).view(32,1)
        reward=reward.view(32,1)
        target_q=reward+temp*target_q
        #mask=self.get_mask(state)
        #valid_next_q=next_q[mask==1]
        #mask2=self.get_mask(next_state)
        loss=F.mse_loss(current_q,target_q.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(0.001*param.data +0.999*target_param.data)
        #return loss   
    def draw_ticket(self,ticket_num,least_ticket_num):
        #only for ticket_num=3 and least_ticket_num=2
        random.shuffle(env.game.ticket_list)
        ticket_draw_pool=[]
        before_draw_ticket_num=len(env.game.ticket_list)
        for i in range(min(ticket_num,len(env.game.ticket_list))):
            ticket_draw_pool.append(env.game.ticket_list.pop())
        ticket_index=[]
        for ticket in ticket_draw_pool:
            ticket_index.append(ticket.index)
        ticket_index.extend([-1]*(3-len(ticket_index)))#in case ticket<=2
        ticket_index.insert(0,before_draw_ticket_num)
        current_state=self.save_current_state(tickets=ticket_index)
        action=self.select_action(current_state)#1-8 action
        if action==1:
            self.tickets.append(ticket_draw_pool[0])
        elif action==2:
            self.tickets.append(ticket_draw_pool[1])
        elif action==3:
            self.tickets.append(ticket_draw_pool[0])
            self.tickets.append(ticket_draw_pool[1])
        elif action==4:
            self.tickets.append(ticket_draw_pool[2])
        elif action==5:
            self.tickets.append(ticket_draw_pool[0])
            self.tickets.append(ticket_draw_pool[2])
        elif action==6:
            self.tickets.append(ticket_draw_pool[1])
            self.tickets.append(ticket_draw_pool[2])
        elif action==7:
            self.tickets.append(ticket_draw_pool[0])
            self.tickets.append(ticket_draw_pool[1])
            self.tickets.append(ticket_draw_pool[2])
        env.game.ticket_list.extend([ticket for ticket in ticket_draw_pool if ticket not in self.tickets])
        for ticket in self.tickets:
            if pathfinder(self.dist_matrix,ticket.cities[0],ticket.cities[1])[0]<99:
                self.finished_tickets.append(ticket)
        self.status=-1
    def sample(self):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, self.batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    def Turn(self):
        _log('Turn:',env.game.turn)
        _log('Player:',self.name)
        self.status=0
        reward=-0.01#encourage quick end
        while True:
            current_state=self.save_current_state()
            if self.status==-1:
                _log("DDQN Action Done")
                if config.LEARN:
                    self.learn()
                    """loss=self.learn()
                    if env.game.end_turn!=999:
                        with open('Learn loss.txt','a') as file:
                            file.write(str(loss)+'\n')"""
                return
            elif self.status==0:
                action=self.select_action(current_state)
                self.status=action
            elif self.status==1:
                action=self.select_action(current_state)
                if action==0:
                    for i in range(len(env.game.show_pile)):
                        if env.game.show_pile[i].color==0:
                            self.cards.append(env.game.show_pile.pop(i))
                            if len(env.game.draw_pile)>0:
                                env.game.show_pile.append(env.game.draw_pile.pop())
                            env.game.draw_pile_update()
                            env.game.rainbowupdate()
                            break
                    self.status=-1#waiting
                elif action==9:
                    self.draw_random_card()
                    self.status=1.5
                elif 1<=action<=8:
                    for i in range(len(env.game.show_pile)):
                        if env.game.show_pile[i].color==action:
                            self.cards.append(env.game.show_pile.pop(i))
                            if len(env.game.draw_pile)>0:
                                env.game.show_pile.append(env.game.draw_pile.pop())
                            env.game.draw_pile_update()
                            env.game.rainbowupdate()
                            break
                    self.status=1.5
            elif self.status==1.5:
                action=self.select_action(current_state)
                if action==9:
                    self.draw_random_card()
                    self.status=-1
                elif 1<=action<=8:
                    for i in range(len(env.game.show_pile)):
                        if env.game.show_pile[i].color==action:
                            self.cards.append(env.game.show_pile.pop(i))
                            if len(env.game.draw_pile)>0:
                                env.game.show_pile.append(env.game.draw_pile.pop())
                            env.game.draw_pile_update()
                            env.game.rainbowupdate()
                            break
                    self.status=-1
            elif self.status==2:
                action=self.select_action(current_state)
                route=env.game.route_list[action]
                if route.color==9:
                    self.status=2.5
                    self.to_buy_route_index=action
                else:
                    r_cost=can_buy(self,route)
                    reward+=route.scoredict[route.length]
                    complete_buy_route(self,route,r_cost)
                    self.status=-1
            elif self.status==2.5:#only after status=2
                action=self.select_action(current_state)
                route=env.game.route_list[self.to_buy_route_index]
                player_have=self.cards_visualizer()
                r_cost=[0,0,0,0,0,0,0,0,0]
                r_cost[action]=min(player_have[action],route.length)
                r_cost[0]=route.length-r_cost[action]
                reward=route.scoredict[route.length]
                complete_buy_route(self,route,r_cost)
                self.status=-1
            elif self.status==3:
                self.draw_ticket(3,1)
            elif self.status==3.5:
                self.draw_ticket(3,2)
            next_state=self.save_current_state()
            self.memory.append((current_state,action,reward,next_state,self.done))
            if env.game.end_turn!=999:
                self.done=1
                reward+=self.calc_end_bonus_score()
