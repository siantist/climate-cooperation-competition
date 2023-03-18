# function for each region to rank their importances <= take from actual facts?
def region_rank_importances(self):
  max_map, min_map, av_map = all_constants_calc(self)
  all_constants = self.all_constants
  feat_list = ['xA_0', 'xK_0', 'xL_0', 'xL_a', 'xa_1', 'xa_2', 'xa_3', 'xdelta_A', 'xg_A', 'xgamma', 'xl_g', 'xsigma_0']
  region_ranked_scores ={}
  for region_key in all_constants:
      region_map = all_constants[region_key]
      # convert to a score (between 0 and 1)?
      ranked_scores= {}
      for feat_key in feat_list:
          feat_val = region_map[feat_key]
          av_val = av_map[feat_key]
          max_val = max_map[feat_key]
          min_val = min_map[feat_key]
          # calculate based on max and min and average ?
          scaled_score = np.abs((feat_val - min_val)/(max_val - min_val))
          ranked_scores[feat_key] = scaled_score

      region_ranked_scores[region_key] = ranked_scores

  return region_ranked_scores
    
  
  
# calculate maximum, minimum, average of the self all Constants
def all_constants_calc(self):
    all_constants = self.all_constants
    const_map = {}
    feat_list = ['xA_0', 'xK_0', 'xL_0', 'xL_a', 'xa_1', 'xa_2', 'xa_3', 'xdelta_A', 'xg_A', 'xgamma', 'xl_g', 'xsigma_0']
    for region_map in all_constants:
        for feat_key in feat_list:
            feat_val = region_map[feat_key]
            ca = const_map.get(feat_key)
            if ca == None:
                const_map[feat_key] = [feat_val]
            else:
                current_array = const_map[feat_key]
                current_array.append(feat_val)
                const_map[feat_key] = current_array
    max_map = {}
    min_map = {}
    av_map = {}
    for feat_key in feat_list:
        feat_val = feat_list[feat_key]
        fmax = np.max(feat_val)
        fmin = np.min(feat_val)
        fav = np.mean(feat_val)
        max_map[feat_key] = fmax
        min_map[feat_key] = fmin
        av_map[feat_key] = fav

    return max_map, min_map, av_map 

def wt_from_rating(ratings):
  Ind=0
  Tot= sum(ratings)
  n = len(ratings)
  Wts =[]
  for r in ratings:
       p = r/10
       Wts.append(p) 
  return Wts 

# function to convert the weight into a sampled action matrix ? 
# based on def generate_action_masks 
def sample_weights_action_mask(self, wts_dict, num_regions):
  mask_dict = {region_id: None for region_id in range(self.num_regions)}
  for i in range(self.num_regions):
    wts_i = wts_dict[i]
    #sample each action
    mask = self.default_agent_action_mask.copy()
    if self.negotiation_on:
      # sample the action 
      for j in range(self.len_actions):
        action01 = np.random.choice([1,0], p =[wts_i[j], 1- wts_i[j]])

# exploitation step with this
def linsumassignment_step(self, actions=None):

    for key in self.keys_to_optimize:
        costmat = self.global_state[key]["value"][self.timestep]
        # print the cost mat
        print("current key is:", key)
        print("cost mat is:", costmat)
        res = fh.minimize(costmat)
        # use the result to assign the jobs

        self.track_values_key[key].append(res)

        ar = self.track_values_key[key]
        # set global state
        self.set_global_state(key, np.array(ar), self.timestep)

    obs = self.generate_observation()
    rew = {region_id: 0.0 for region_id in range(self.num_regions)}
    done = {"__all__": 0}
    info = {}
    return obs, rew, done, info
  
