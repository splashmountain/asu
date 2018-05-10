#import geopandas as gpd
import pysal as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from heapq import heappush, heappop
import pulp
#from pylab import figure, scatter, show
#import networkx as nx
#import pylab


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


def gen_region(radius,seed,qW):
	regions={}
	regions[0] = seed
	rings={}
	rings[0] = seed
    
	for i in range(1, radius+1):
		neigh = []
		donut = rings[i-1]
		for j in donut:
			neigh.extend(qW.neighbors[j])
		neigh=list(set(neigh))
		donut=[fruit for fruit in neigh if fruit not in regions[i-1]]
		rings[i] = donut
		region=[]
		region.extend(regions[i-1])
		region.extend(donut)
		regions[i]= region
	return regions,rings	

#given an order, create a dictionary for its previous order neighborhood (around a seed set)
#where key is each area in the given order, value is its previous order queen neighbors
#it is used to create contiguous constraints, will be called many times as needed
def gen_prev_order(cur_order,rings,qW):
	prev_neigh = {}
	for i in rings[cur_order]:
		prev_neigh[i] = set(qW.neighbors[i]) & set(rings[cur_order-1])
	return prev_neigh

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

def mip(radius,seed,mdf_dict,qW):
	regions,rings=gen_region(radius,seed,qW)
	region = regions[radius]

#######################################################
############ 1. create the model ######################
#######################################################
	my_lp_problem = pulp.LpProblem("My LP Problem", pulp.LpMaximize) 

#######################################################
############ 2. create decision variables #############
#######################################################
	decision_variables ={}
	for i in region:
		variablestr = str('x' + str(i))
		variable = pulp.LpVariable(str(variablestr), lowBound = 0, upBound = 1, cat= 'Integer')
		decision_variables[i] = variable 

#######################################################
############ 3. create objective function #############
#######################################################
	total_pop = ""

	for key, value in decision_variables.items():
		#for rownum, row in mdf.iterrows():
		#	if key == rownum:
		#		total_pop += row['POP']*value
		total_pop += mdf_dict[key]['POP']*value
            
	my_lp_problem += total_pop
	#print ("Optimization function: "+str(total_pop))  


#################################################
############ 4. create constraints ##############
#################################################

########### 4.1 create constraint such that the seeds should be included.
	for i in seed:
		my_lp_problem += (decision_variables[i] == 1)

	#my_lp_problem += (decision_variables[114] ==1) 

	#overlap=list(set(region) & set(areas_in_c24))
	#for j in overlap:
	#	my_lp_problem += (decision_variables[j] == 0) # to avoid overlapping

########## 4.2 create constraint such that ur is no less than the threshold.
	nume = ""
	denomi = ""
	for key, value in decision_variables.items():
		#for rownum, row in mdf.iterrows():
		#	if key == rownum:
		#		nume += row['UNEMP']*value
		#		denomi += (row['UNEMP']+row['EMP'])*value
		nume += mdf_dict[key]['UNEMP']*value
		denomi += (mdf_dict[key]['UNEMP']+mdf_dict[key]['EMP'])*value
	my_lp_problem += (nume >= threshold*denomi)

########## 4.3 create constraint such that the region is contiguous.
	for i in range(1,radius+1):
		prev=gen_prev_order(i,rings,qW)
		for j in prev: # key of the dict
			self_var = decision_variables[j]
			neigh = ""
			for va in prev[j]: #for each value of the key
				neigh += decision_variables[va]
			my_lp_problem += (self_var <= neigh)
    
###########################################
############ 5. run optimization ##########
###########################################

	optimization_result = my_lp_problem.solve()
	assert optimization_result == pulp.LpStatusOptimal
	print("Status:", pulp.LpStatus[my_lp_problem.status])
	print("Optimal Solution to the problem: ", pulp.value(my_lp_problem.objective))
	
	results=[]
	for v in my_lp_problem.variables():
 		#print(v.name, "=", v.varValue) 
 		if v.varValue == 1.0:
 			results.append(int(v.name[1:]))

	return results

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
	f.suptitle("Figure 3: Region 3")
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
	

	seed=clusters[18]
	radius = 25

	#seed=clusters[2]
	#radius = 6

	#areas_in_c2=[110, 132, 133, 135, 366, 407, 426, 427, 43, 44, 45, 463, 48, 529, 530, 560, 91]
	#areas_in_c4=[152, 159, 163, 164, 167, 168, 201, 202, 245, 257, 445, 505, 506, 515, 517, 63, 68]
	#areas_in_c24=areas_in_c2+areas_in_c4

	# let's solve it
	results = mip(radius,seed,mdf_dict,qW)
	#results = mip(radius,seed,mdf_dict,qW,common)
			
	r_pop,r_ur=compute_region_stats(results,mdf_dict)			

	# when seed is clusters[2], i.e.,[45, 133, 132, 530]
	#print(results) #[43, 44, 45, 48, 91, 110, 132, 133, 135, 366, 377, 407, 426, 427, 457, 463, 529, 530, 560]
	#print(r_pop) #98424
	#print(r_ur) #0.0649042633193

	# when seed is clusters[0], i.e.,[18, 572, 21, 461, 540]
	#print(results) #[17, 18, 19, 21, 45, 48, 77, 79, 80, 97, 98, 120, 121, 122, 131, 132, 133, 146, 147, 149, 198, 229, 230, 232, 328, 366, 406, 424, 427, 461, 462, 463, 512, 540, 543, 549, 571, 572, 579]
	#print(r_pop) #179341
	#print(r_ur) #0.0663497623217413

	# when seed is clusters[4], i.e.,[68, 245, 164, 505, 152]
	print(results) #[63, 68, 152, 155, 159, 163, 164, 168, 201, 202, 245, 257, 445, 505, 506, 515, 517]
	print(r_pop) #66305
	print(r_ur) #0.0652754803410103


	#geoid=get_geoid(mdf,results)
	#plot_region(geoid,df,shp_path)


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



#clusters[0] (r=6)
#[105, 115, 119, 120, 121, 122, 131, 147, 149, 17, 18, 19, 204, 21, 214, 218, 224, 232, 233, 250, 28, 280, 289, 290, 31, 328, 338, 461, 462, 512, 540, 543, 549, 571, 572, 579, 58, 77, 79, 80, 97, 98]
#180762
#0.0649384485117


#clusters[2] (r=6)
#[110, 132, 133, 135, 366, 407, 426, 427, 43, 44, 45, 463, 48, 529, 530, 560, 91]
#85144
#0.0649549806121

#265906
#clusters[4] (r=5)
#[152, 159, 163, 164, 167, 168, 201, 202, 245, 257, 445, 505, 506, 515, 517, 63, 68]
#65744
#0.0649938296997

