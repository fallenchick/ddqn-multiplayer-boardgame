"""Baseline opponents, also written from scratch.

* ``AIPlayer``    — heuristic decision-tree policy.  Values every route by its
  own score plus the expected reduction in shortest-path distance weighted by
  ticket value, then greedily maximises expected gain.
* ``GithubPlayer`` — third-party open-source baseline agent, kept as an
  independent reference point for evaluation.
* ``RandomPlayer`` — uniform over legal actions.
"""

import random

import numpy as np

from scipy.sparse.csgraph import dijkstra

from . import config, env
from .env import Player, Route, can_buy, complete_buy_route, path_to_route_tuple, pathfinder

def _log(*args, **kwargs):
    """Debug output, silenced unless ``config.VERBOSE`` is on.

    The original script printed every turn, which dominated the runtime of a
    long training run and buried the actual results.  Routing all of it
    through this helper makes the default output clean and quiet.
    """
    if config.VERBOSE:
        print(*args, **kwargs)




# ========================================================================
# AIPlayer — heuristic decision-tree opponent
# ========================================================================

class AIPlayer(Player):
    def draw_ticket(self,ticket_num,least_ticket_num):#选择VP/需要总长最大值
        random.shuffle(env.game.ticket_list)
        raw_candidate_list=[[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]#if ticket_num>3,use green codes above to get raw_candidate_list
        candidate_list=[]
        for choice in raw_candidate_list:
            if len(choice)>=least_ticket_num:
                candidate_list.append(choice)
        candidate_len=[]
        candidate_value=[]
        ticket_draw_pool=[]
        for i in range(min(ticket_num,len(env.game.ticket_list))):
            ticket_draw_pool.append(env.game.ticket_list.pop())
        for choice in candidate_list:
            candidate_tickets=[]
            tickets_value=0
            for i in range(len(choice)):
                candidate_tickets.append(ticket_draw_pool[i])
                tickets_value+=ticket_draw_pool[i].value
            routes_len,_=self.need_routes_finder(candidate_tickets)
            candidate_len.append(routes_len)
            candidate_value.append(tickets_value)
        for i in range(len(candidate_list)):
            if candidate_len[i]>self.remain_route_len-2:#2 is arbitrary
                candidate_value[i]=-999
            elif candidate_len[i]==0:
                candidate_value[i]=candidate_value[i]/100#make choice with len=0 less likely to happen
            else:
                if candidate_len[i]<=self.remain_route_len:
                    candidate_value[i]=candidate_value[i]/candidate_len[i]
                else:
                    candidate_value[i]=-candidate_value[i]#when no positive return of taking tickets, minimize loss
        max_value=max(candidate_value)
        max_i=candidate_value.index(max_value)
        choice=candidate_list[max_i]
        _log(self.name,'take',len(choice),'tickets')
        for num in range(len(choice)):
            self.tickets.append(ticket_draw_pool[num])
            #_log('AI takes ticket:')
            #self.tickets[-1].print_ticket()
        for ticket in self.tickets:
            if pathfinder(self.dist_matrix,ticket.cities[0],ticket.cities[1])[0]<99:
                self.finished_tickets.append(ticket)
        env.game.ticket_list.extend([ticket for ticket in ticket_draw_pool if ticket not in self.tickets])
                
    def need_routes_finder(self,tickets):#for AI
        #tickets=[ticket for ticket in self.tickets if ticket not in self.finished_tickets]
        original_dist_matrix=env.game.dist_matrix.copy()
        for route in self.routes:
            a,b=route.cities
            original_dist_matrix[a][b]=1e-4
            original_dist_matrix[b][a]=1e-4
        if len(tickets)==0:
            return (0,[])
        elif len(tickets)==1:
            current_dist_matrix=original_dist_matrix.copy()
            path=pathfinder(current_dist_matrix,tickets[0].cities[0],tickets[0].cities[1])
            return path[0],path_to_route_tuple(path[1])
        elif len(tickets)==2:
            minimum=999
            min_path=[(0,0)]*100
            for i,j in [(0,1),(1,0)]:
                current_dist_matrix=original_dist_matrix.copy()
                first_path=pathfinder(current_dist_matrix,tickets[i].cities[0],tickets[i].cities[1])
                assumed_connected=path_to_route_tuple(first_path[1])
                for cities in assumed_connected:
                    a,b=cities
                    current_dist_matrix[a][b]=1e-4
                    current_dist_matrix[b][a]=1e-4
                second_path=pathfinder(current_dist_matrix,tickets[j].cities[0],tickets[j].cities[1])
                temp_path=path_to_route_tuple(first_path[1])+path_to_route_tuple(second_path[1])
                if first_path[0]+second_path[0]<minimum:#and len(temp_path)<len(min_path):#can change to minimized turns(longer routes),but no good effect <=minimum
                    minimum=first_path[0]+second_path[0]
                    min_path=temp_path
            return minimum,list(set(min_path))
        elif len(tickets)==3:
            minimum=999
            min_path=[(0,0)]*100
            for i,j,k in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
                current_dist_matrix=original_dist_matrix.copy()
                first_path=pathfinder(current_dist_matrix,tickets[i].cities[0],tickets[i].cities[1])
                assumed_connected=path_to_route_tuple(first_path[1])
                for cities in assumed_connected:
                    a,b=cities
                    current_dist_matrix[a][b]=1e-4
                    current_dist_matrix[b][a]=1e-4
                second_path=pathfinder(current_dist_matrix,tickets[j].cities[0],tickets[j].cities[1])
                assumed_connected=path_to_route_tuple(second_path[1])
                for cities in assumed_connected:
                    a,b=cities
                    current_dist_matrix[a][b]=1e-4
                    current_dist_matrix[b][a]=1e-4
                third_path=pathfinder(current_dist_matrix,tickets[k].cities[0],tickets[k].cities[1])
                temp_path=path_to_route_tuple(first_path[1])+path_to_route_tuple(second_path[1])+path_to_route_tuple(third_path[1])
                if first_path[0]+second_path[0]+third_path[0]<minimum:#and len(temp_path)<len(min_path):
                    minimum=first_path[0]+second_path[0]+third_path[0]
                    min_path=temp_path
            return minimum,list(set(min_path))
        #_log(len(tickets))
    def get_route_value(self):
        assumed_connected_path=self.need_routes_tuples
        assumed_dist_matrix=env.game.dist_matrix.copy()
        owned_route_tuples=[]
        route_len_dict={}
        route_value=[0]*len(env.game.route_list)#get route value per len
        for i in range(len(env.game.route_list)):
            route=env.game.route_list[i]
            route_value[i]=route.scoredict[route.length]/route.length
            route_len_dict[route.cities]=route.length
        for route in self.routes:
            a,b=route.cities
            assumed_dist_matrix[a][b]=1e-4
            assumed_dist_matrix[b][a]=1e-4
            owned_route_tuples.append(route.cities)
        for city_pair in assumed_connected_path:
            a,b=city_pair
            assumed_dist_matrix[a][b]=1e-4
            assumed_dist_matrix[b][a]=1e-4
        tickets=[ticket for ticket in self.tickets if ticket not in self.finished_tickets]
        for ticket in tickets:
            path=pathfinder(assumed_dist_matrix,ticket.cities[0],ticket.cities[1])
            ticket_need_routes=path_to_route_tuple(path[1])
            uncompleted_ticket_need_routes=[city_pair for city_pair in ticket_need_routes if city_pair not in owned_route_tuples]
            need_routes_totallen=0
            for city_pair in uncompleted_ticket_need_routes:
                need_routes_totallen+=route_len_dict[city_pair]
            for i in range(len(env.game.route_list)):
                route=env.game.route_list[i]
                if route.cities in uncompleted_ticket_need_routes:
                    route_value[i]+=2*ticket.value/need_routes_totallen#2*ticket.value*route.length/need_routes_totallen/route.length
        value_dict={}
        for i in range(len(env.game.route_list)):
            value_dict[env.game.route_list[i]]=route_value[i]
        return value_dict
    def Turn(self):
        _log('Turn:',env.game.turn)
        _log('Player:',self.name)
        tickets=[ticket for ticket in self.tickets if ticket not in self.finished_tickets]
        self.need_routes_tuples=self.need_routes_finder(tickets)[1]
        for route in self.routes:
            if route.cities in self.need_routes_tuples:
                self.need_routes_tuples.remove(route.cities)
        if len(self.need_routes_tuples)==0:
            if self.remain_route_len<=5 or len(env.game.ticket_list)<3:#5 is an arbitrary num
                #try to end game quick
                can_buy_list=[]
                for route in env.game.route_list:
                    if can_buy(self,route):
                        can_buy_list.append(route)
                if len(can_buy_list)==0:#may draw wild card,but under this algorithm unnecessary
                    self.draw_random_card()
                    self.draw_random_card()
                else:
                    to_buy_route=Route('0000000')
                    for route in can_buy_list:
                        if route.length>to_buy_route.length:
                            to_buy_route=route
                    r_cost=can_buy(self,route)
                    if route.rid[5]=='9':
                        player_have=self.cards_visualizer()
                        i=player_have.index(max(player_have[1:]))
                        r_cost=[0,0,0,0,0,0,0,0,0]
                        r_cost[i]=min(player_have[i],route.length)
                        r_cost[0]=route.length-r_cost[i]
                    complete_buy_route(self,route,r_cost)
            else:
                self.draw_ticket(3,1)
                return
        else:#focus on what first 2(pairs of) routes need
            route_value_dict=self.get_route_value()
            sorted_value_dict=dict(sorted(route_value_dict.items(),key=lambda item:item[1],reverse=True))
            focuses=list(sorted_value_dict)[:4]
            for focusroute in focuses:
                if can_buy(self,focusroute):
                    r_cost=can_buy(self,focusroute)
                    if focusroute.rid[5]=='9':
                        player_have=[0,0,0,0,0,0,0,0,0]
                        for card in self.cards:
                            player_have[card.color]+=1
                        i=player_have.index(max(player_have[1:]))
                        r_cost=[0,0,0,0,0,0,0,0,0]
                        r_cost[i]=min(player_have[i],focusroute.length)
                        r_cost[0]=focusroute.length-r_cost[i]
                    complete_buy_route(self,focusroute,r_cost)
                    return
            #cannot buy focusroute
            focusroutes_need=[0,0,0,0,0,0,0,0,0]
            for focusroute in focuses:
                if focusroute.color!=9:
                    focusroutes_need[focusroute.color]+=focusroute.length
            for card in self.cards:
                focusroutes_need[card.color]-=1
            for i in range(2):#will not draw wild card
                drawn=0
                for card in env.game.show_pile:
                    if len(self.need_routes_tuples)==1:
                        if card.color==0:
                            self.cards.append(card)
                            env.game.show_pile.remove(card)
                            if len(env.game.draw_pile)>0:
                                env.game.show_pile.append(env.game.draw_pile.pop())
                            env.game.draw_pile_update()
                            env.game.rainbowupdate()
                            _log(self.name,'drawn color',card.color,'from show pile')
                            return
                    if focusroutes_need[card.color]>0:
                        self.cards.append(card)
                        drawn+=1
                        env.game.show_pile.remove(card)
                        if len(env.game.draw_pile)>0:
                            env.game.show_pile.append(env.game.draw_pile.pop())
                        env.game.draw_pile_update()
                        env.game.rainbowupdate()
                        _log(self.name,'drawn color',card.color,'from show pile')
                        break
                if not drawn:
                    self.draw_random_card()
                    _log(self.name,'drawn random card')


# ========================================================================
# GithubPlayer — third-party open-source baseline
# ========================================================================

class GithubPlayer(Player):
    def need_routes_finder(self):#for AI
        original_dist_matrix=env.game.dist_matrix.copy()
        for route in self.routes:
            a,b=route.cities
            original_dist_matrix[a][b]=1e-4
            original_dist_matrix[b][a]=1e-4
        current_dist_matrix=original_dist_matrix.copy()
        need_routes_tuples=[]
        tickets=[ticket for ticket in self.tickets if ticket not in self.finished_tickets]
        for ticket in tickets:
            path=pathfinder(current_dist_matrix,ticket.cities[0],ticket.cities[1])
            for tup in path_to_route_tuple(path[1]):
                if tup not in need_routes_tuples:
                    need_routes_tuples.append(tup)
        return need_routes_tuples
    def draw_ticket(self,ticket_num,least_ticket_num):
        random.shuffle(env.game.ticket_list)
        ticket_draw_pool=[]
        for i in range(min(ticket_num,len(env.game.ticket_list))):
            ticket_draw_pool.append(env.game.ticket_list.pop())
        self.tickets.append(ticket_draw_pool[0])
        self.tickets.append(ticket_draw_pool[1])#already random
        for ticket in self.tickets:
            if pathfinder(self.dist_matrix,ticket.cities[0],ticket.cities[1])[0]<99:
                self.finished_tickets.append(ticket)
        env.game.ticket_list.extend([ticket for ticket in ticket_draw_pool if ticket not in self.tickets])
    def Turn(self):
        _log('Turn:',env.game.turn)
        _log('Player:',self.name)
        tickets=[ticket for ticket in self.tickets if ticket not in self.finished_tickets]
        self.need_routes_tuples=self.need_routes_finder()
        for route in self.routes:
            if route.cities in self.need_routes_tuples:
                self.need_routes_tuples.remove(route.cities)
        can_buy_list=[]
        for route in env.game.route_list:
            if can_buy(self,route):
                can_buy_list.append(route)
        can_buy_list=sorted(can_buy_list,key=lambda route:route.length,reverse=True)
        if len(self.need_routes_tuples)==0:
            if env.game.turn<=40 and len(env.game.ticket_list)>=2:
                self.draw_ticket(3,1)
            elif max(self.cards_visualizer())>5 and len(can_buy_list)>0:
                to_buy_route=can_buy_list[0]
                r_cost=can_buy(self,to_buy_route)
                if to_buy_route.rid[5]=='9':
                    player_have=[0,0,0,0,0,0,0,0,0]
                    for card in self.cards:
                        player_have[card.color]+=1
                    i=player_have.index(max(player_have[1:]))
                    r_cost=[0,0,0,0,0,0,0,0,0]
                    r_cost[i]=min(player_have[i],to_buy_route.length)
                    r_cost[0]=to_buy_route.length-r_cost[i]
                _log(r_cost)
                complete_buy_route(self,to_buy_route,r_cost)
            else:
                if env.game.turn<=80:
                    annealing_factor=(1-env.game.turn/200)*80+20
                else:
                    annealing_factor=(1-env.game.turn/200)*100
                if random.randint(1,100)>=annealing_factor and len(can_buy_list)>0:
                    to_buy_route=can_buy_list[0]
                    r_cost=can_buy(self,to_buy_route)
                    if to_buy_route.rid[5]=='9':
                        player_have=[0,0,0,0,0,0,0,0,0]
                        for card in self.cards:
                            player_have[card.color]+=1
                        i=player_have.index(max(player_have[1:]))
                        r_cost=[0,0,0,0,0,0,0,0,0]
                        r_cost[i]=min(player_have[i],to_buy_route.length)
                        r_cost[0]=to_buy_route.length-r_cost[i]
                    complete_buy_route(self,to_buy_route,r_cost)
                else:
                    for card in env.game.show_pile:
                        if card.color==0:
                            self.cards.append(card)
                            env.game.show_pile.remove(card)
                            if len(env.game.draw_pile)>0:
                                env.game.show_pile.append(env.game.draw_pile.pop())
                            env.game.draw_pile_update()
                            env.game.rainbowupdate()
                            return
                    self.draw_random_card()
                    self.draw_random_card()
        else:
            to_buy_route=None
            for route in can_buy_list:
                if route.cities in self.need_routes_tuples:
                    to_buy_route=route
                    break
            if to_buy_route!=None:
                r_cost=can_buy(self,to_buy_route)
                if to_buy_route.rid[5]=='9':
                    player_have=[0,0,0,0,0,0,0,0,0]
                    for card in self.cards:
                        player_have[card.color]+=1
                    i=player_have.index(max(player_have[1:]))
                    r_cost=[0,0,0,0,0,0,0,0,0]
                    r_cost[i]=min(player_have[i],to_buy_route.length)
                    r_cost[0]=to_buy_route.length-r_cost[i]
                complete_buy_route(self,to_buy_route,r_cost)
            else:
                for card in env.game.show_pile:
                    if card.color==0:
                        self.cards.append(card)
                        env.game.show_pile.remove(card)
                        if len(env.game.draw_pile)>0:
                            env.game.show_pile.append(env.game.draw_pile.pop())
                        env.game.draw_pile_update()
                        env.game.rainbowupdate()
                        return
                self.draw_random_card()
                self.draw_random_card()
