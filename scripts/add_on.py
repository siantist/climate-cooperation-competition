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
  
