import geopandas as gpd
import pysal as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from heapq import heappush, heappop
#import pulp
#from pylab import figure, scatter, show
#import networkx as nx
#import pylab

# astar part is based on the code by Julien Rialland (https://github.com/jrialland)

threshold = 0.0649


def gen_mdf(attr_path, dataframe):
	df=pd.read_csv(attr_path,dtype={'geoid': str})
	df.drop(['record', 'st_fips', 'cnty_fips', 'tract_fips', 'name'], axis=1, inplace=True)

	cols = ['pop', 'emp']
	df[cols] = df[cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',',''), errors='coerce'))
	df['sum'] = df['unemp'] + df['emp']
	df['ur'] = df['unemp']/(df['unemp'] + df['emp'])
	df.loc[df['sum'] == 0, 'ur'] = 0
	df.columns = map(str.upper, df.columns)

	mdf=pd.merge(dataframe, df, on="GEOID")
	return mdf, df 

def gen_mdf_dict(mdf):
	#{0: {'EMP': 3638, 'POP': 6813, 'UNEMP': 77},
 	#1: {'EMP': 1393, 'POP': 3515, 'UNEMP': 75},
 	#2: {'EMP': 4053, 'POP': 7987, 'UNEMP': 205},
	new_mdf=mdf[['POP','EMP','UNEMP']].copy()
	mdf_dict=new_mdf.to_dict('index')
	return mdf_dict

def genSeeds(mdf,qW):
	asu_seeds = mdf.loc[mdf['UR'] >= threshold] # find rows whose ur >= 0.065
	asu_seeds=asu_seeds.index.tolist()

	queens = {} # a sub-dictionary of w_queen.neighbors, this sub-dict only contains asu's queens
	for seed in asu_seeds:
		queens[seed]=list(set(qW.neighbors[seed]) & set(asu_seeds))
        
	graph=queens
	clusters=[]
	visited=[]
	path=[]
	for i in graph:
		if i not in visited:
			q=[i]
			path=[]
			while q:
				v = q.pop()
				if v not in path:
					path += [v]
					q += graph[v]
			visited.extend(path)        
			clusters.append(path)
	return clusters


def compute_region_stats(region,mdf_dict):
	cur_pop = 0
	cur_ur = 0
	nume = 0 
	denomi = 0
	for i in region:
		cur_pop += mdf_dict[i]['POP']
		nume += mdf_dict[i]['UNEMP']
		denomi += (mdf_dict[i]['UNEMP'] + mdf_dict[i]['EMP'])
	cur_ur = nume/denomi
	return cur_pop, cur_ur

def is_goal_reached(region,goal,mdf_dict):
	#there are two aspects of the goal: pop and ur
	goal_pop = goal[0]
	goal_ur = goal[1]

	cur_pop, cur_ur = compute_region_stats(region,mdf_dict)
	return cur_ur >= goal_ur and cur_pop >= goal_pop


def f_value(region,source,goal,mdf_dict):
	#right now f_value is like dijkstra's score
	# only have g_value's
	start_pop = source[0]
	start_ur = source[1]
	#goal_pop = goal[0]
	#goal_ur = goal[1]
    
    # compute the pop and ur for the current region
	cur_pop, cur_ur = compute_region_stats(region,mdf_dict)
	
	#h = (goal_pop - cur_pop) /goal_pop +  (goal_ur - cur_ur) /goal_ur  
	

	# if the ur is less than the threshold, then its f_value is penalized
	penality = 0
	if cur_ur < threshold:
		penality = 1000
	g = (start_pop - cur_pop) /start_pop +  (start_ur - cur_ur) /start_ur  
	return g + penality

#Given a region, return its neighbors
def neighbors_poly(region,qW):
	neigh = []
	for i in region:
		neigh.extend(qW.neighbors[i])
	#neigh = [qW.neighbors[i] for i in region]
	neigh=list(set(neigh))
	donut=[fruit for fruit in neigh if fruit not in region]
	return donut


# start is a seed region as a set, 
# goal is a tuple, first item is goal_pop on that seed region, second item is goal_ur on that seed region
# In this version, we ignore the path, only caring about the resulting states
def astar(start, source, goal,mdf_dict,qW):
	#searchNodes dictionary: {node: [0:f,1:node]}
	searchNodes = {}
	start=tuple(sorted(start))
	if is_goal_reached(start, goal,mdf_dict):
		return start

	startNode = searchNodes[start] = [f_value(start, source, goal,mdf_dict), start]
        
	openSet = [] # aka frontier [0:f,1:node]
	heappush(openSet, startNode)
	while openSet:
		current = heappop(openSet)
		if is_goal_reached(current[1],goal,mdf_dict):
			return current[1]
           
		for n in neighbors_poly(current[1],qW):
			# add each node/state in the searhNodes dictionary on the fly
			#if n in areas_in_c2:
			#	continue
			new = tuple(sorted(current[1] + (n,)))
			if new in searchNodes:
				continue
			# here neighbor means neighboring state not neighboring area

			neighbor = searchNodes[new] = [f_value(new, source, goal,mdf_dict), new]
                
			heappush(openSet, neighbor)
	return None

def get_geoid(mdf,results):
	rdf=mdf.loc[results]
	geoid = rdf['GEOID'].tolist()
	return geoid

def prep_gdf(shp_path,df):
	ind = 'GEOID'
	shp_data = gpd.read_file(shp_path)
	gdf = shp_data.merge(df, on=ind)
	return gdf

def plot_region(geoid,df,shp_path):
	index = 'GEOID'
	gdf = prep_gdf(shp_path,df)
	gdf = gdf.set_index(index)
	# Setup figure
	f, ax = plt.subplots(1)
	# Plot base layer of polygons
	gdf.plot(ax=ax, facecolor='k', linewidth=0.1,alpha=0.6)
	# Plot seeds
	neis = gdf.loc[geoid, :]
	neis.plot(ax=ax, facecolor='red', linewidth=0,alpha=0.6)
	# Title
	f.suptitle("Figure 1: Region 1")
	# Style and display on screen
	plt.axis('equal')
	#ax.set_ylim(388000, 393500)
	#ax.set_xlim(336000, 339500)
	plt.show()


def go():
	shp_path = 'data/tl_2016_49_tract.shp'
	attr_path = 'data/UT_asu_exampleData.csv'

	qW = ps.queen_from_shapefile(shp_path)
	dataframe = ps.pdio.read_files(shp_path)

	mdf,df = gen_mdf(attr_path, dataframe)

	mdf_dict = gen_mdf_dict(mdf)

	clusters = genSeeds(mdf,qW)
	
	#areas_in_c2 = [43, 44, 45, 48, 91, 110, 132, 133, 135, 366, 377, 407, 426, 427, 457, 463, 529, 530, 560]

	seeds=clusters[2]
	start = tuple(seeds)
	#source = [6832,0.0859972]
	
	
	s_pop,s_ur=compute_region_stats(start,mdf_dict)
	source = [s_pop,s_ur]
	#goal = [15053,threshold]

	
	goal = [99000,threshold] #goal for clusters[2]
	#goal = [190762,threshold] ##goal for clusters[0]
	#goal = [170000,threshold] ##goal for clusters[0]
	#goal = [175000,threshold] ##goal for clusters[0]
	#goal = [65744,threshold] #goal for clusters[4]

	# let's solve it
	#results = list(astar(start, source, goal,mdf_dict,qW,areas_in_c2))
	results = list(astar(start, source, goal,mdf_dict,qW))	
	r_pop,r_ur=compute_region_stats(results,mdf_dict)			

	# when seed is clusters[2], i.e.,[45, 133, 132, 530]
	#print(results) #[43, 44, 45, 48, 91, 110, 132, 133, 135, 366, 377, 407, 426, 427, 457, 463, 529, 530, 560]
	#print(r_pop) #98424
	#print(r_ur) #0.0649042633193

	# when seed is clusters[0], i.e.,[18, 572, 21, 461, 540]
	print(results) #[17, 18, 19, 21, 45, 48, 77, 79, 80, 97, 98, 120, 121, 122, 131, 132, 133, 146, 147, 149, 198, 229, 230, 232, 328, 366, 406, 424, 427, 461, 462, 463, 512, 540, 543, 549, 571, 572, 579]
	print(r_pop) #179341
	print(r_ur) #0.0663497623217413

	# when seed is clusters[4], i.e.,[68, 245, 164, 505, 152]
	#print(results) #[63, 68, 152, 155, 159, 163, 164, 168, 201, 202, 245, 257, 445, 505, 506, 515, 517]
	#print(r_pop) #66305
	#print(r_ur) #0.0652754803410103
	
	#results=[17, 18, 19, 21, 45, 48, 77, 79, 80, 97, 98, 120, 121, 122, 131, 132, 133, 
	#146, 147, 149, 198, 229, 230, 232, 328, 366, 406, 424, 427, 461, 462, 463, 512, 540, 543, 549, 571, 572, 579]

	geoid=get_geoid(mdf,results)
	plot_region(geoid,df,shp_path)


def go2():
	shp_path = 'data/tl_2016_49_tract.shp'
	attr_path = 'data/UT_asu_exampleData.csv'

	qW = ps.queen_from_shapefile(shp_path)
	dataframe = ps.pdio.read_files(shp_path)

	mdf,df = gen_mdf(attr_path, dataframe)
	mdf_dict = gen_mdf_dict(mdf)

	clusters = genSeeds(mdf,qW)
	
	for c in clusters:
		start = tuple(c)
		s_pop,s_ur=compute_region_stats(start,mdf_dict)
		source = [s_pop,s_ur]

		goal = [177992,threshold] 

		# let's solve it
		results = list(astar(start, source, goal,mdf_dict,qW))
			
		r_pop,r_ur=compute_region_stats(results,mdf_dict)	



if __name__ == '__main__':
	# To activate this environment, use:
	# > source activate geopandas-test-mac
	# To deactivate an active environment, use:
	# > source deactivate

	go()
