import copy
import numpy as np

class Network:
    def __init__(self):
        self.net = {}
        self.nodes = []
        self.edges = []
    def add_nodes(self,nodes):
        keys = list(self.net.keys())
        nodes = [n for n in nodes if n not in keys]
        for node in nodes:
            self.net[node] = []
            self.nodes.append(node)
    def add_edges(self,edges):
        edges = [edges] if type(edges) == tuple else edges
        for (u,v) in edges:
            if u not in self.nodes:
                self.net[u] = []
                self.nodes.append(u)
            if v not in self.nodes:
                self.net[v] = []
                self.nodes.append(v)
            self.net[u].append(v)
            self.net[v].append(u)
            self.edges.append((u,v))
    def remove_edge(self,edge):
        del self.edges[self.edges.index(edge)]
        use1 = self.net[edge[0]]
        del use1[use1.index(edge[1])]
        self.net[edge[0]] = use1

        use2 = self.net[edge[1]]
        del use2[use2.index(edge[0])]
        self.net[edge[1]] = use2
    def copy(self):
        return copy.deepcopy(self)
    def neighbors(self,node):
        return self.net[node]
    

num_neurons = 15
num_beginners = 4
num_evolutions = 40
num_children_rate = 0.6
num_random_survivors = 2
learning_rate = 1
num_children_surviving = 10
begin_evolution_change = 15
mutation_rate = 1
num_connections = 1
min_signal_strength = 0.01
oper_lb = -30
oper_hb = -oper_lb
stop_range = 0.05
delta_loss = 0.2
gsum = []
symbols = [None,"+","-","*","/"]
x = []
y = []

opers = np.linspace(oper_lb,oper_hb,num_neurons-num_beginners).tolist()
nodes = [str(i) for i in opers]
    # Add nodes
def generate_network():
    G = Network()
    G.add_nodes(["Final", "Initial"]+nodes)
    connections = []
    for i in nodes:
        for j in range(num_connections):
            c = nodes[np.random.randint(0,len(nodes))]
            connections.append((i,c))
    G.add_edges(connections)
    for i in range(num_beginners):
        c = nodes[np.random.randint(0,len(nodes))]
        G.add_edges((c,"Initial"))
        d = nodes[np.random.randint(0,len(nodes))]
        G.add_edges((d,"Final"))
    return G

def fforward(tokens,network):
    for token in tokens:
        forward(network, float(token),1,"Initial",["Initial"],"Initial")

def forward(network, signal, loss, node,traveled,output):
    global gsum,min_signal_strength,delta_loss
    if abs(signal) <= min_signal_strength or loss < 0:
        return 0
    if node == "Final":
        gsum.append(signal)
        return 0
    
    neighbors = [i for i in list(network.neighbors(node))]# if i not in traveled]
    if node == "Initial":
        node = "0"
    for neigh in neighbors:
        u = traveled
        u.append(node)
        forward(network,(signal+float(node))*loss,loss-delta_loss,neigh,u,output+","+str(neigh))

def evolve(changes, network):
    
    for i in range(changes):
        edges = list(network.edges)
        if np.random.rand() <= mutation_rate and edges:
            use = edges[np.random.randint(0,len(edges))]
            while use[0] == "Initial" or use[1] == "Initial" or use[0] == "Final" or use[1] == "Final":
                use = edges[np.random.randint(0,len(edges))]
            network.remove_edge((use[0],use[1]))
        c = nodes[np.random.randint(0,len(nodes))]
        d = nodes[np.random.randint(0,len(nodes))]
        network.add_edges((c,d))
    return network

def prepare(expr):
    almost = tokenize(expr)
    for i in range(len(almost)):
        if almost[i] in symbols:
            almost[i] = symbols.index(almost[i])
    return almost

def tokenize(expr):
    tokens = []
    num = ""
    for ch in expr:
        if ch.isdigit():
            num += ch
        else:
            if num:
                tokens.append(num)
                num = ""
            if ch.strip():
                tokens.append(ch)
    if num:
        tokens.append(num)
    return tokens

def train(expression,networks):
    global num_children,x,y,gsum
    stack = begin_evolution_change/learning_rate
    best_networks = networks
    best_networks_results = []
    for i in range(num_evolutions):
        stack = int(stack * learning_rate)
        children = []
        fav_child = best_networks[0]
        if len(best_networks_results)>0:
            fav_child = best_networks[best_networks_results.index(min(best_networks_results))]
        for j in best_networks:
            children += [evolve(stack, j.copy()) for k in range(int(num_neurons*num_children_rate))]
        children += [fav_child]
        output = []
        kkk = 1
        for child in children:
            fforward(prepare(expression),child)
            output.append(sum(gsum))
            kkk+=1
            gsum=[]
        results = [abs(float(eval(expression))-i) for i in output]
        copy_results = list(results)
        copy_children = list(children)
        
        if abs(min(copy_results)) < stop_range:
            return [copy_children[copy_results.index(min(copy_results))]]
                
        j = 0
        surviving_children = []
        surviving_children_results = []
        while j < num_children_surviving and copy_results and copy_children:
            lowest_index = copy_results.index(min(copy_results))
            surviving_children_results.append(min(copy_results))
            surviving_children.append(copy_children[lowest_index])
            if len(copy_results) > 0:
                del copy_results[lowest_index]
            if len(copy_children) > 0:
                del copy_children[lowest_index] 
            j+=1
        for j in range(num_random_survivors):
            if len(copy_results) > 0 and len(copy_children) > 0:
                r = np.random.randint(0,len(copy_results))
                surviving_children.append(copy_children[r])
                surviving_children_results.append(copy_results[r])
                del copy_results[r]
                del copy_children[r]
            
        best_networks = surviving_children
        best_networks_results = surviving_children_results
    return best_networks


def test(x,network):
    global gsum
    i = tokenize(x)
    for j in range(len(i)):
        if i[j] in symbols:
            i[j] = symbols.index(i[j])
    
    fforward(i,network)
    return sum(gsum)
    
a = []
t = []
for i in range(10):
    print("Episode " + str(i+1))
    gsum = []
    n = generate_network()
    best_networks = train("2",[n])
    res0 = test("2",best_networks[0])
    res = test("3",best_networks[0])
    a.append(res)
    t.append(res0)
a = np.array(a)
t = np.array(t)
print("Two average: " + str(np.average(t)))
print("Two STD: " + str(np.std(t)))
print()
print("Three average: " + str(np.average(a)))
print("Three STD: " + str(np.std(a)))
