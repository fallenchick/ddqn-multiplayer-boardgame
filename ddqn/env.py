"""Game environment for *Railroad Tour* (铁路环游).

Contains the board game itself: cities, routes, tickets, the deck, the turn
state machine, and the Dijkstra helper used for route valuation.

All players interact through a single shared instance exposed as ``env.game``.
The environment is fully self-contained — the map, route table and ticket deck
are embedded in :meth:`Game.specialsetup`, so there is nothing to download in
order to run this project.
"""

import random
from collections import deque

import numpy as np
from scipy.sparse.csgraph import dijkstra

from . import config
from .config import PLAYER_NUM

def _log(*args, **kwargs):
    """Debug output, silenced unless ``config.VERBOSE`` is on.

    The original script printed every turn, which dominated the runtime of a
    long training run and buried the actual results.  Routing all of it
    through this helper makes the default output clean and quiet.
    """
    if config.VERBOSE:
        print(*args, **kwargs)




# ========================================================================
# Shortest-path helpers
# ========================================================================

def create_distance_matrix(routes, city_num=None):
    """All-pairs shortest-path matrix over the currently *unowned* routes.

    ``city_num`` defaults to the shared game's city count.  Passing it
    explicitly keeps the helper usable before a game has been published.
    """
    if city_num is None:
        city_num = game.city_num
    num_nodes = city_num+1
    dist_matrix = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist_matrix, 0)
    for route in routes:
        city1, city2, dist = route.cities[0],route.cities[1],route.length
        dist_matrix[city1, city2] = dist
        dist_matrix[city2, city1] = dist
    return dist_matrix
def pathfinder(dist_matrix,start_city,end_city):
    distances, predecessors = dijkstra(dist_matrix, directed=False, return_predecessors=True,indices=start_city)
    path = []
    current_city = end_city
    while current_city != start_city:
        path.append(current_city)
        if current_city==-9999:
            return 9999,[]
        current_city = predecessors[current_city]
    path.append(start_city)
    path.reverse()
    return int(distances[end_city]),path
def path_to_route_tuple(path):
    tuples=[]
    for i in range(len(path)-1):
        if path[i]<path[i+1]:
            tuples.append((path[i],path[i+1]))
        else:
            tuples.append((path[i+1],path[i]))
    return tuples



# ========================================================================
# Card / Ticket
# ========================================================================

class Card(object):
    def __init__(self,cid):
        self.cid=cid#1-110,color0:rainbow 1:pink 2:blue 3:green 4:yellow 5:orange 6:red 7:white 8:black 9:any
        if 1<=self.cid<=14:
            self.color=0
        else:
            self.color=(self.cid-3)//12
class Ticket(object):
    def __init__(self,tid):#'203222'-From 20(New York) to 32(Seattle) value=22'
        self.tid=tid
        self.cities=(int(tid[:2]),int(tid[2:4]))
        self.value=int(tid[4:])
    def print_ticket(self):
        _log('From',game.city_name_dict[self.cities[0]],'to',game.city_name_dict[self.cities[1]],'Reward:',self.value)


# ========================================================================
# Route
# ========================================================================

class Route(object):
    def __init__(self,rid):
        self.rid=rid #'0116520-From 01(Atlanta) to 16(Miami) 5 2(Blue) 0 specialdigit'
        self.cities=(int(rid[:2]),int(rid[2:4]))
        self.length=int(rid[4])
        self.color=int(rid[5])
        self.scoredict={1:1,2:2,3:4,4:7,5:10,6:15,8:21}
        self.special_digit=int(rid[6])
        self.controller=0#None
def can_buy(player,route):
    player_have=player.cards_visualizer()
    if player.remain_route_len<route.length or route.controller!=0:
        return False
    if player_have[0]+max(player_have[1:]) >=route.length and route.color==9:#buy process Moved to Action2
        route_cost=[route.length,9,0,0,0,0,0,0,0]
        return route_cost
    else:
        if route.color!=9:
            if player_have[0]+player_have[route.color]>=route.length:
                route_cost=[0,0,0,0,0,0,0,0,0]
                route_cost[route.color]=min(player_have[route.color],route.length)
                route_cost[0]=route.length-player_have[route.color]
                return route_cost
    return False
def complete_buy_route(player,route,r_cost):
    while sum(r_cost)>0:
        for card in player.cards:
            if r_cost[card.color]>0:
                game.discard_pile.append(card)
                player.cards.remove(card)
                r_cost[card.color]-=1
    player.routes.append(route)
    route.controller=game.players.index(player)+1
    if len(game.players)<4:
        for temproute in game.route_list:
            if temproute.cities==route.cities and temproute.controller==0:
                temproute.controller=-1
    player.remain_route_len-=route.length
    _log('Successfully bought route color',route.color,'from',game.city_name_dict[route.cities[0]],'to',game.city_name_dict[route.cities[1]])
    _log('Player remain route len:',player.remain_route_len)
    remain_route_list=[route for route in game.route_list if route.controller==0]
    game.dist_matrix=create_distance_matrix(remain_route_list)
    for ticket in player.tickets:
        if pathfinder(player.dist_matrix,ticket.cities[0],ticket.cities[1])[0]<99:
            player.finished_tickets.append(ticket)
    if player.remain_route_len<=2:
        _log('Last turn!')
        if game.end_turn==999:
            game.end_turn=game.turn+len(game.players)


# ========================================================================
# Player (base class)
# ========================================================================

class Player():
    def __init__(self,name='test'):
        self.name=name
        self.tickets=[]
        self.cards=[]
        self.routes=[]
        self.dist_matrix=create_distance_matrix(self.routes)
        self.need_routes_tuples=[]#for AI
        self.finished_tickets=[]
        self.remain_route_len=45
        _log('Created player:',self.name)
        for i in range(4):
            self.draw_random_card()
        #self.draw_ticket(3,2)#may change on gametype
    def cards_visualizer(self):
        temp=[0,0,0,0,0,0,0,0,0]
        for card in self.cards:
            temp[card.color]+=1
        return temp
    def draw_random_card(self):
        if len(game.draw_pile)>0:
            self.cards.append(game.draw_pile.pop())
            game.draw_pile_update()
    def calc_route_score(self):#add end_score
        current_route_score=0
        for route in self.routes:
            current_route_score+=route.scoredict[int(route.rid[4])]
        return current_route_score
    def calc_end_bonus_score(self):
        bonus_score=0
        for ticket in self.tickets:
            if ticket in self.finished_tickets:
                bonus_score+=ticket.value
            else:
                bonus_score-=ticket.value
        #add longest_route
        return bonus_score
    def calc_end_score(self):
        current_route_score=self.calc_route_score()
        bonus_score=self.calc_end_bonus_score()
        return current_route_score+bonus_score


# ========================================================================
# HumanPlayer
# ========================================================================

class HumanPlayer(Player):
    def __init__(self,name='test'):
        super().__init__(name)
        self.print_held_card()
    def print_held_card(self):
        _log('Held cards:')
        _log(self.cards_visualizer())
    def draw_show_card(self,chances):
        game.print_show_pile()
        if len(game.show_pile)>0:
            i=-1
            while i not in range(len(game.show_pile)):
                i=int(input("Draw which card? 0 1 2 3 4"))
                if i in range(len(game.show_pile)):#take using label
                    color=game.show_pile[i].color
                    if color!=0 or chances>=2:
                        self.cards.append(game.show_pile.pop(i))
                        game.show_pile.append(game.draw_pile.pop())
                        game.draw_pile_update()
                        game.rainbowupdate()
                        self.print_held_card()
                        break
                    else:
                        _log('cannot draw')
                        return 0
                else:
                    _log('cannot draw!')
                    return 0
            if color==0:
                return 2
            else:
                return 1

    def Action1(self):
        chances=2
        while chances>0:
            i=input('Draw random card or draw specific card? 1/2')
            if i=='1':
                self.draw_random_card()
                self.print_held_card()
                chances-=1
            elif i=='2':
                chances=chances-self.draw_show_card(chances)
            else:
                _log('Illegal input')
            _log('chances'+str(chances))
        _log('Action done')
        
    def Action2(self):
        while True:
            input_rid=input('buy which route?')
            matched=0
            for route in game.route_list:
                if route.rid==input_rid:
                    matched=1
                    break
            if not matched:
                _log('cannot buy nonexisted route!')
                continue
            temproute=Route('0000000')
            for temproute in self.routes:
                if temproute.cities==route.cities:
                    break
            if temproute.cities==route.cities:
                _log('cannot buy repeated route!')
                continue
            r_cost=can_buy(self,route)
            if r_cost:
                if route.rid[5]=='9':
                    player_have=self.cards_visualizer()
                    i=int(input('use which color? 1-8'))
                    if 1<=i<=8 and player_have[0]+player_have[i]>=route.length:
                        r_cost=[0,0,0,0,0,0,0,0,0]
                        r_cost[i]=min(player_have[i],route.length)
                        r_cost[0]=route.length-r_cost[i]
                    else:
                        _log('cannot buy the route using this color!')
                        continue
                #else:r_cost=r_cost
            else:
                _log('cannot purchase!')
                continue
            complete_buy_route(self,route,r_cost)
            self.print_held_card()
            _log('Action done')
            return
    def Action3(self):
        self.draw_ticket(3,1)
        _log('Action done')
    def draw_ticket(self,ticket_num,least_ticket_num):#add long_route ticket pool
        random.shuffle(game.ticket_list)
        _log('Choose at least',least_ticket_num,'tickets')
        ticket_draw_pool=[]
        for i in range(min(ticket_num,len(game.ticket_list))):
            ticket_draw_pool.append(game.ticket_list.pop())
        for ticket in ticket_draw_pool:
            ticket.print_ticket()
        while True:
            inputstr=input('Take which tickets?use numbers 012... etc.')
            if least_ticket_num<=len(inputstr)<=ticket_num and inputstr.isdigit():
                temp=[]
                for c in inputstr:
                    temp.append(int(c))
                if max(temp)<ticket_num:
                    for num in temp:
                        if 0<=num<ticket_num:
                            self.tickets.append(ticket_draw_pool[num])
                            _log('Take ticket:')
                            self.tickets[-1].print_ticket()
                    game.ticket_list.extend([ticket for ticket in ticket_draw_pool if ticket not in self.tickets])
                    for ticket in self.tickets:
                        if pathfinder(self.dist_matrix,ticket.cities[0],ticket.cities[1])[0]<99:
                            self.finished_tickets.append(ticket)
                    return
                else:
                    _log('Invalid input')
            else:
                _log('Invalid input')
    def Turn(self):
        _log('Turn:',game.turn)
        _log('Player:',self.name)
        game.print_show_pile()
        i=0
        while i not in [1,2,3]:
            i=int(input('Which move?1-3'))
            if i==1:
                self.Action1()
            elif i==2:
                self.Action2()
            elif i==3:
                self.Action3()
            else:
                _log('Invalid input! type 1-3')


# ========================================================================
# Game
# ========================================================================

class Game(object):
    def __init__(self):
        self.specialsetup()
        temp_card_list=[]
        for i in range(1,111):
            temp_card_list.append(Card(i))
        random.shuffle(temp_card_list)
        self.draw_pile=temp_card_list.copy()
        self.discard_pile=[]
        self.show_pile=[]
        for i in range(5):
            self.show_pile.append(self.draw_pile.pop())
        self.print_show_pile()
        self.end_turn=999
        self.turn=0
    def specialsetup(self,gametype=1):
        if gametype==1:
            self.city_num=36
            self.city_name_dict={1:'Atlanta',2:'Boston',3:'Calgary',4:'Charleston',5:'Chicago',6:'Dallas',
                                 7:'Denver',8:'Duluth',9:'El Paso',10:'Helena',11:'Houston',12:'Kansas City',
                                 13:'Las Vegas',14:'Little Rock',15:'Los Angeles',16:'Miami',17:'Montreal',18:'Nashville',
                                 19:'New Orleans',20:'New York',21:'Oklahoma City',22:'Omaha',23:'Phoenix',24:'Pittsburgh',
                                 25:'Portland',26:'Raleigh',27:'Saint Louis',28:'Salt Lake City',29:'San Francisco',30:'Santa Fe',
                                 31:'Sault Ste. Marie',32:'Seattle',33:'Toronto',34:'Vancouver',35:'Washington',36:'Winnipeg'}
            self.ticket_list=[Ticket('012917'),Ticket('011709'),Ticket('012006'),
                              Ticket('021612'),Ticket('032313'),Ticket('032807'),
                              Ticket('053009'),Ticket('051907'),Ticket('051516'),
                              Ticket('062011'),Ticket('070904'),Ticket('072411'),
                              Ticket('080910'),Ticket('081108'),Ticket('101508'),
                              Ticket('111205'),Ticket('113612'),Ticket('143611'),
                              Ticket('152021'),Ticket('151620'),Ticket('153209'),
                              Ticket('163310'),Ticket('171913'),Ticket('173420'),
                              Ticket('182517'),Ticket('183108'),Ticket('203222'),
                              Ticket('213109'),Ticket('232511'),Ticket('303413')]
            self.TICKET_NUM=len(self.ticket_list)
            for ticket in self.ticket_list:
                ticket.index=self.ticket_list.index(ticket)
            #7th digit reserved for Mega Europe(xx city1 xx city2 x length x color0-9 x special_digit 0left 1 2needlocomotive1-2 3tunnel 9right)
            self.route_list=[Route('0126290'),Route('0126299'),Route('0118190'),Route('0119430'),Route('0119450'),Route('0116520'),Route('0104290'),
                             Route('0217290'),Route('0217299'),Route('0220240'),Route('0220260'),
                             Route('0334390'),Route('0332490'),Route('0310490'),Route('0336670'),
                             Route('0426290'),Route('0416410'),
                             Route('0508360'),Route('0522420'),Route('0527230'),Route('0527270'),Route('0524350'),Route('0524380'),Route('0533470'),
                             Route('0621290'),Route('0621299'),Route('0609460'),Route('0611190'),Route('0611199'),Route('0614290'),
                             Route('0710430'),Route('0728360'),Route('0728340'),Route('0723570'),Route('0730290'),Route('0721460'),Route('0712480'),Route('0712450'),Route('0722410'),
                             Route('0836480'),Route('0810650'),Route('0822290'),Route('0822299'),Route('0833610'),Route('0831390'),
                             Route('0923390'),Route('0915680'),Route('0911630'),Route('0921540'),Route('0930290'),
                             Route('1032640'),Route('1028310'),Route('1022560'),Route('1036420'),
                             Route('1119290'),
                             Route('1222190'),Route('1222199'),Route('1221290'),Route('1221299'),Route('1227210'),Route('1227220'),
                             Route('1315290'),Route('1328350'),
                             Route('1421290'),Route('1419330'),Route('1418370'),Route('1427290'),
                             Route('1529340'),Route('1529310'),Route('1523390'),
                             Route('1619660'),
                             Route('1731580'),Route('1733390'),Route('1720320'),
                             Route('1826380'),Route('1824440'),Route('1827290'),
                             Route('2024270'),Route('2024230'),Route('2035250'),Route('2035280'),
                             Route('2130320'),
                             Route('2330390'),
                             Route('2433290'),Route('2427530'),Route('2426290'),Route('2435290'),
                             Route('2532190'),Route('2532199'),Route('2529530'),Route('2529510'),Route('2528620'),
                             Route('2635290'),Route('2635299'),
                             Route('2829550'),Route('2829570'),
                             Route('3136690'),Route('3133290'),
                             Route('3234190'),Route('3234199')]
        if gametype==2:
            self.city_num=47
            self.city_name_dict={1:'Amsterdam',2:'Ankara',3:'Athens',4:'Barcelona',5:'Berlin',6:'Brest',
                                 7:'Brindisi',8:'Brussels',9:'Bucarest',10:'Budapest',11:'Cadiz',12:'Constantinople',
                                 13:'Danzig',14:'Dieppe',15:'Edinburgh',16:'Erzurum',17:'Essen',18:'Frankfurt',
                                 19:'Izmir',20:'Kharkov',21:'Kiev',22:'Copenhagen',23:'Lisbon',24:'London',
                                 25:'Madrid',26:'Marseille',27:'Moscow',28:'Munich',29:'Palermo',30:'Pamplona',
                                 31:'Paris',32:'Petrograd',33:'Riga',34:'Rome',35:'Rostov',36:'Sarajevo',
                                 37:'Sevastopol',38:'Smolensk',39:'Sochi',40:'Sofia',41:'Stockholm',42:'Venice',
                                 43:'Vienna',44:'Vilnius',45:'Warsaw',46:'Zagrab',47:'Zurich'}
            self.ticket_list=[Ticket('153107'),Ticket('154607'),Ticket('050908'),Ticket('022010'),Ticket('193408'),
                              Ticket('254708'),Ticket('142508'),Ticket('034411'),Ticket('414311'),Ticket('020305'),
                              Ticket('020513'),Ticket('022420'),Ticket('022521'),Ticket('022714'),Ticket('023117'),
                              Ticket('023411'),Ticket('024310'),Ticket('030511'),Ticket('032416'),Ticket('032516'),
                              Ticket('032714'),Ticket('033113'),Ticket('033406'),Ticket('034308'),Ticket('052407'),
                              Ticket('052513'),Ticket('052712'),Ticket('053107'),Ticket('053409'),Ticket('054303'),
                              Ticket('242510'),Ticket('242719'),Ticket('243103'),Ticket('243410'),Ticket('244310'),
                              Ticket('252725'),Ticket('253107'),Ticket('253410'),Ticket('254313'),Ticket('273118'),
                              Ticket('273417'),Ticket('274312'),Ticket('313410'),Ticket('314308'),Ticket('344306')]
            self.TICKET_NUM=len(self.ticket_list)
            for ticket in self.ticket_list:
                ticket.index=self.ticket_list.index(ticket)
            self.route_list=[]
    def draw_pile_update(self):
        if len(self.draw_pile)==0 and len(self.discard_pile)>0:
            self.draw_pile=self.discard_pile.copy()
            self.discard_pile=[]
            random.shuffle(self.draw_pile)
            _log('Draw pile reshuffle!')
    def rainbowupdate(self):
        rainbowcnt=0
        for card in self.show_pile:
            if card.color==0:
                rainbowcnt+=1
        loop_num=0
        while rainbowcnt>=3:#可能出现牌堆全是彩卡
            loop_num+=1
            _log('Too many wild cards!')
            for card in self.show_pile:
                self.discard_pile.append(card)
            self.show_pile=[]
            for i in range(5):
                if len(self.draw_pile)>0:
                    self.show_pile.append(self.draw_pile.pop())
                    self.draw_pile_update()
            rainbowcnt=0
            for card in self.show_pile:
                if card.color==0:
                    rainbowcnt+=1
            if loop_num>10:
                return
    def get_show_pile(self):
        temp=[-1,-1,-1,-1,-1]
        for i in range(len(self.show_pile)):
            temp[i]=self.show_pile[i].color
        return temp
    def print_show_pile(self):
        _log('Current show pile:')
        temp=self.get_show_pile()
        _log(temp)
# The single shared game instance.  Every player reads and writes this
# object; `train.py` (re)publishes it via `env.game = Game()`.
#
# NOTE: it is deliberately created *without* `specialsetup()` here, because
# `Game.__init__` leaves the route table empty and the distance matrix cannot
# be built until `specialsetup()` has populated the board.
game = None
